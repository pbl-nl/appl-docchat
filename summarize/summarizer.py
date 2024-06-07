import os
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans
from langchain_core.prompts import PromptTemplate
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
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, llm_type=None, llm_model_type=None) -> None:
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
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type

        # get llm object
        self.llm = LLMCreator(self.llm_type,
                              self.llm_model_type).get_llm()

    def get_centroids(self, embeddings: np.array) -> np.ndarray:
        """
        gets the initial centroids of the embedding clusters

        Parameters
        ----------
        embeddings : np.array
            the embeddings of the chunks

        Returns
        -------
        np.ndarray
            ndarray of shape (n_clusters, n_features)
        """
        kmeans = KMeans(n_clusters=settings.SUMMARIZER_CENTROIDS, random_state=42)
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_

        return centroids

    def nearest_neighbors(self, centroid: np.ndarray, collection: List) -> List[Tuple[int, np.ndarray]]:
        """
        Find the nearest neighbors to the centroids of the clusters of the embeddings

        Parameters
        ----------
        centroid : np.ndarray
            numpy array representing the centroid for which we want to find the nearest neighbors
        collection : list
            the vector database collection

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            list of tuples with first element the chunk id and second element the chunk text
        """
        distances = []

        embeddings = np.array(collection['embeddings'])
        for i in range(len(embeddings)):
            chunk_ids = collection['ids'][i]
            chunk_text = collection['documents'][i]
            embedding = collection['embeddings'][i]
            distance = ut.cosine_similarity(embedding, centroid)
            distances.append((chunk_ids, chunk_text, distance))
        # sort result on distance
        distances.sort(key=lambda x: x[2])
        # get ids and text of closest chunks
        nearest_chunks = [(distance[0], distance[1]) for distance in distances][:settings.SUMMARIZER_CENTROID_NEIGHBORS]

        return nearest_chunks

    def get_chunks(self, collection: List) -> List[str]:
        """
        Gets the chunks that are nearest to the centroids of the clusters

        Parameters
        ----------
        collection : list
            the vector database collection

        Returns
        -------
        List[str]
            list of chunk texts, sorted on chunk id
        """
        # extract embeddings from vector store
        embeddings = np.array(collection['embeddings'])
        # find cluster centroids
        centroids = self.get_centroids(embeddings)
        # find text pieces near the cluster centroids
        chunk_tuples = []
        chunk_ids_processed = []
        for centroid in centroids:
            nearest_chunks = self.nearest_neighbors(centroid, collection)
            for nearest_chunk in nearest_chunks:
                chunk_ids, chunk_text = nearest_chunk
                # make sure that the indices are unique
                if chunk_ids not in chunk_ids_processed:
                    chunk_ids_processed.append(chunk_ids)
                    chunk_tuples.append((int(chunk_ids), chunk_text))

        chunk_tuples.sort(key=lambda x: x[0])
        chunk_texts_sorted = [chunk_tuple[1] for chunk_tuple in chunk_tuples]

        return chunk_texts_sorted

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
        first = True
        for chunk_text in chunk_texts:
            if first:
                summarize_prompt_template = PromptTemplate.from_template(template=pr.SUMMARY_TEMPLATE)
                summarize_prompt = summarize_prompt_template.format(text=chunk_text)
                summary = self.llm.invoke(summarize_prompt).content
                first = False
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

            # get ids and texts for all chunks from the vector database, sorted by the numeric id
            chunk_texts = []
            for idx in range(len(collection['ids'])):
                idx_text = collection['documents'][idx]
                chunk_texts.append(idx_text)

            # if refine method is chosen
            if self.summary_method == "Refine":
                # summarize by looping over all chunks
                summary = self.get_summary(chunk_texts)
            # else if Map Reduce method is chosen
            elif self.summary_method == "Map_Reduce":
                # first select chunks and join them
                chunk_texts = self.get_chunks(collection)
                chunks_joined = '\n\nNext Summary Piece: '.join(chunk_texts)
                # then summarize
                summarize_prompt_template = PromptTemplate.from_template(template=pr.SUMMARY_MAPREDUCE_TEMPLATE)
                summarize_prompt = summarize_prompt_template.format(chunks_joined=chunks_joined)
                summary = self.llm.invoke(summarize_prompt).content
            # else if Hybrid method is chosen
            elif self.summary_method == "Hybrid":
                # first select chunks
                chunk_texts = self.get_chunks(collection)
                # summarize by looping over all selected chunks
                summary = self.get_summary(chunk_texts)

            # define output file and store summary
            file_name, _ = os.path.splitext(file)
            result = os.path.join(self.content_folder, "summaries", str(file_name) + "_" +
                                  str.lower(self.summary_method) + ".txt")
            with open(file=result, mode="w", encoding="utf8") as f:
                f.write(summary)
