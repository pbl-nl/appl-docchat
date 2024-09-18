import os
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans
from langchain_core.prompts import PromptTemplate
from loguru import logger
# local imports
import settings
from ingest.ingester import Ingester
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator
from query.llm_creator import LLMCreator
import prompts.prompt_templates as pr
import utils as ut


class Summarizer:
    """
    When the summarizer class parameters are read from settings.py, object is initiated without parameter settings
    When parameters are read from GUI, object is initiated with parameter settings listed
    """
    def __init__(self, collection_name: str, content_folder: str, vecdb_folder: str, summary_method: str,
                 embeddings_provider=None, retriever_type=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, llm_provider=None, llm_model=None) -> None:
        """
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
        """
        load_dotenv()
        self.collection_name = collection_name
        self.summary_method = summary_method
        self.content_folder = content_folder
        self.vecdb_folder = vecdb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.llm_provider = settings.LLM_PROVIDER if llm_provider is None else llm_provider
        self.llm_model = settings.LLM_MODEL if llm_model is None else llm_model

        # get llm object
        self.llm = LLMCreator(self.llm_provider,
                              self.llm_model).get_llm()

    def get_centroids(self,
                      chunk_embeddings_and_texts: List[Tuple[List, str]]) -> np.ndarray:
        """
        gets the initial centroids of the embedding clusters

        Parameters
        ----------
        chunk_embeddings_and_texts : List[Tuple[List, str]]
            list of tuples with chunk embedding and text

        Returns
        -------
        np.ndarray
            ndarray of shape (n_clusters, n_features)
        """
        kmeans = KMeans(n_clusters=settings.SUMMARIZER_CENTROIDS, random_state=42)
        chunk_embeddings = np.array([chunk_embedding_and_text[0] for chunk_embedding_and_text in
                                     chunk_embeddings_and_texts], dtype=float)

        kmeans.fit(chunk_embeddings)
        centroids = kmeans.cluster_centers_

        return centroids

    def nearest_neighbors(self,
                          centroid: np.ndarray,
                          chunk_embeddings_and_texts: List[Tuple[List, str]]) -> List[Tuple[int, np.ndarray]]:
        """
        Find the nearest neighbors to the centroids of the clusters of the embeddings

        Parameters
        ----------
        centroid : np.ndarray
            numpy array representing the centroid for which we want to find the nearest neighbors
        chunk_embeddings_and_texts : List[Tuple[List, str]]
            list of tuples with chunk embedding and text

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            list of tuples with first element the chunk id and second element the chunk text
        """
        distances = []
        for idx, chunk_embedding_and_text in enumerate(chunk_embeddings_and_texts):
            parent_chunk_embedding = np.array(chunk_embedding_and_text[0], dtype=float)
            parent_chunk_text = chunk_embedding_and_text[1]
            distance = ut.cosine_similarity(parent_chunk_embedding, centroid)
            distances.append((idx, parent_chunk_text, distance))

        # sort result on distance
        distances.sort(key=lambda x: x[2])
        # get ids and text of closest chunks
        nearest_chunks = [(distance[0], distance[1]) for distance in distances][:settings.SUMMARIZER_CENTROID_NEIGHBORS]

        return nearest_chunks

    def get_file_chunks_embeddings_and_texts(self, collection: List) -> List[str]:
        """
        Gets all the chunks from tthat are nearest to the centroids of the clusters

        Parameters
        ----------
        collection : list
            the vector database collection

        Returns
        -------
        List[str]
            list of chunk texts, sorted on chunk id
        """
        # get ids and texts for all chunks from the vector database, sorted by the numeric id
        chunk_embeddings_and_texts = []
        for idx in range(len(collection['ids'])):
            if self.retriever_type != "parent":
                chunk_embedding = collection['embeddings'][idx]
                chunk_text = collection['documents'][idx]
                chunk_embeddings_and_texts.append((chunk_embedding, chunk_text))
            else:
                # parent_chunk_embedding is stored as a string, so convert to list
                chunk_embedding = collection['metadatas'][idx]['parent_chunk_embedding'].split(",")
                chunk_text = collection['metadatas'][idx]["parent_chunk"]
                if chunk_text not in [chunk_embedding_and_text[1] for chunk_embedding_and_text in
                                      chunk_embeddings_and_texts]:
                    chunk_embeddings_and_texts.append((chunk_embedding, chunk_text))

        return chunk_embeddings_and_texts

    def get_nearest_chunks(self, collection: List) -> List[str]:
        """
        Gets the chunks that are nearest to the centroids of the clusters

        Parameters
        ----------
        collection : list
            the vector database collection

        Returns
        -------
        List[str]
            list of chunk texts
        """
        # Extract embeddings directly from vector store
        chunk_embeddings_and_texts = self.get_file_chunks_embeddings_and_texts(collection)

        # find cluster centroids based on embeddings
        centroids = self.get_centroids(chunk_embeddings_and_texts)

        # find text pieces near the cluster centroids
        chunk_ids_processed = []
        chunk_texts = []
        for centroid in centroids:
            nearest_chunks = self.nearest_neighbors(centroid, chunk_embeddings_and_texts)
            for nearest_chunk in nearest_chunks:
                chunk_id, chunk_text = nearest_chunk
                # make sure that the indices are unique
                if chunk_id not in chunk_ids_processed:
                    chunk_ids_processed.append(chunk_id)
                    chunk_texts.append(chunk_text)

        return chunk_texts

    def get_summary(self, chunk_texts: List[str]) -> str:
        """
        creates a summary from a chunk text, using different prompts based on whether the chunk text is the
        first text or an addition to a previously created summary

        Parameters
        ----------
        chunk_texts : List[str]
            _description_

        Returns
        -------
        str
            The obtained summary
        """
        for i, chunk_text in enumerate(chunk_texts):
            logger.info(f"summarizing chunk {i} of {len(chunk_texts)}")
            if i == 0:
                summarize_prompt_template = PromptTemplate.from_template(template=pr.SUMMARY_TEMPLATE)
                summarize_prompt = summarize_prompt_template.format(text=chunk_text)
                summary = self.llm.invoke(summarize_prompt).content
            else:
                refine_prompt_template = PromptTemplate.from_template(template=pr.REFINE_TEMPLATE)
                refine_prompt = refine_prompt_template.format(summary=summary, text=chunk_text)
                summary = self.llm.invoke(refine_prompt).content

        return summary

    def summarize(self) -> None:
        """
        creates summaries of all files in the folder, using the chosen summarization method. One summary per file.
        """
        # create subfolder "summaries" if not existing
        if 'summaries' not in os.listdir(self.content_folder):
            os.mkdir(os.path.join(self.content_folder, "summaries"))

        # create or update vector store for the documents
        ingester = Ingester(collection_name=self.collection_name,
                            content_folder=self.content_folder,
                            vecdb_folder=self.vecdb_folder)
        ingester.ingest()

        # get embeddings object
        embeddings = EmbeddingsCreator(self.embeddings_provider,
                                       self.embeddings_model).get_embeddings()

        # get vector store object
        vector_store = VectorStoreCreator(self.vecdb_type).get_vectorstore(embeddings,
                                                                           self.collection_name,
                                                                           self.vecdb_folder)

        # list of relevant files to summarize
        files_in_folder = ut.get_relevant_files_in_folder(self.content_folder)

        # loop over all files in the folder
        for file in files_in_folder:
            # get all the chunks from the vector store related to the file
            collection = vector_store.get(where={"filename": file},
                                          include=["metadatas", "embeddings", "documents"])

            # get ids and texts for all chunks from the vector database
            chunk_embeddings_and_texts = self.get_file_chunks_embeddings_and_texts(collection)
            chunk_texts = [chunk_embedding_and_text[1] for chunk_embedding_and_text in chunk_embeddings_and_texts]

            # if refine method is chosen
            if self.summary_method == "Refine":
                # summarize by looping over all chunks
                summary = self.get_summary(chunk_texts)
            # else if Map Reduce method is chosen
            elif self.summary_method == "Map_Reduce":
                # first select chunks and join them
                chunk_texts = self.get_nearest_chunks(collection)
                chunks_joined = '\n\nNext Summary Piece: '.join(chunk_texts)
                # then summarize
                summarize_prompt_template = PromptTemplate.from_template(template=pr.SUMMARY_MAPREDUCE_TEMPLATE)
                summarize_prompt = summarize_prompt_template.format(chunks_joined=chunks_joined)
                summary = self.llm.invoke(summarize_prompt).content
            # else if Hybrid method is chosen
            elif self.summary_method == "Hybrid":
                # first select chunks
                chunk_texts = self.get_nearest_chunks(collection)
                # summarize by looping over all selected chunks
                summary = self.get_summary(chunk_texts)

            # define output file and store summary
            file_name, _ = os.path.splitext(file)
            result = os.path.join(self.content_folder, "summaries", str(file_name) + "_" +
                                  str.lower(self.summary_method) + ".txt")
            with open(file=result, mode="w", encoding="utf8") as f:
                f.write(summary)
