import os
from dotenv import load_dotenv
import numpy as np
# local imports
import settings
from ingest.ingester import Ingester
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator
from query.llm_creator import LLMCreator


def initialize_centroids(data, k):
    """
    Randomly initialize centroids
    """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    """
    Assign data points to the nearest centroid
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(data, clusters, k):
    """
    Update centroids as the mean of assigned data points
    """
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids


def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids
    return clusters, centroids


def euclidean_distance(a, b):
    """
    Calculation of euclidian distance between a and b
    """
    return np.sqrt(np.sum((a - b) ** 2))


def nearest_neighbor(data, point):
    """
    Find the nearest neighbor.

    :param data: numpy array of numpy arrays (each sub-array is a data point).
    :param point: numpy array representing the point for which we want to find the nearest neighbor.
    :return: the nearest neighbor and its index in the data.
    """
    nearest_index = None
    min_distance = float('inf')

    for i in range(len(data)):
        distance = euclidean_distance(data[i], point)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    return data[nearest_index], nearest_index


class Summarizer:
    """
    The summarizer class  parameters are read from settings.py, object is initiated without parameter settings
    When parameters are read from GUI, object is initiated with parameter settings listed
    """
    def __init__(self, collection_name: str, content_folder: str, vecdb_folder: str, summary_method: str,
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, local_api_url=None,
                 file_no=None, llm_type=None, llm_model_type=None, azureopenai_api_version=None):
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
        self.local_api_url = settings.API_URL \
            if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.azureopenai_api_version = settings.AZUREOPENAI_API_VERSION \
            if azureopenai_api_version is None and settings.AZUREOPENAI_API_VERSION is not None \
            else azureopenai_api_version

        # get llm object
        self.llm = LLMCreator(self.llm_type,
                              self.llm_model_type,
                              self.local_api_url,
                              self.azureopenai_api_version).get_llm()

    def summarize(self) -> None:
        """
        Creates instances of all parsers, iterates over all files in the folder
        When parameters are read from GUI, object is initiated with parameter settings listed
        """
        # create or update vector store for the documents
        ingester = Ingester(collection_name=self.collection_name,
                            content_folder=self.content_folder,
                            vecdb_folder=self.vecdb_folder)
        ingester.ingest()

        # create subfolder "summaries" if not existing
        if 'summaries' not in os.listdir(self.content_folder):
            os.mkdir(os.path.join(self.content_folder, "summaries"))

        # get embeddings object
        embeddings = EmbeddingsCreator(self.embeddings_provider,
                                       self.embeddings_model,
                                       self.local_api_url,
                                       self.azureopenai_api_version).get_embeddings()

        # get vector store object
        vector_store = VectorStoreCreator(self.vecdb_type).get_vectorstore(embeddings,
                                                                           self.collection_name,
                                                                           self.vecdb_folder)

        # list of files to summarize
        files_in_folder = [f for f in os.listdir(self.content_folder)
                           if (os.path.isfile(os.path.join(self.content_folder, f)) and
                               os.path.splitext(f)[1] in [".docx", ".html", ".md", ".pdf", ".txt"])]

        # loop over all files in the folder
        for _, file in enumerate(files_in_folder):
            file_name, _ = os.path.splitext(file)
            # get all the chunks from the vector store related to the file
            # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            collection = vector_store.get(where={"filename": file}, include=["metadatas", "embeddings", "documents"])

            chunks = []
            # for item in collection.items():
            for idx in range(len(collection['ids'])):
                idx_id = collection['ids'][idx]
                idx_text = collection['documents'][idx]
                # print(f"chunk found with index {idx_id} and text {idx_text}")
                # convert index to int for sorting purposes
                chunks.append((int(idx_id), idx_text))
            # sort the chunks on id
            chunks.sort(key=lambda x: x[0])

            # if refine method is chosen
            if self.summary_method == "Refine":
                first = True
                for idx, text in enumerate(chunks):
                    print(f"REFINE: chunk processed = {idx} of {len(chunks)}")
                    if first:
                        summarize_prompt = f'''
                                          Summarize the following text: {text}.
                                          Only return the summary, no explanation.
                                          '''
                        summary = self.llm.invoke(summarize_prompt)
                        first = False
                    else:
                        refine_prompt = f'''
                                        Given the following summary: {summary}
                                        \nRefine the summary, with the following added information: {text}
                                        \nOnly return the summary, no other text.
                                        '''
                        summary = self.llm.invoke(refine_prompt)
            # else if Map Reduce method is chosen
            elif self.summary_method == "Map_Reduce":
                # extract data from vector store
                embeddings = np.array(collection['embeddings'])
                # cluster (KNN)
                number_of_clusters = settings.SUMMARIZER_CENTROIDS
                _, centroids = kmeans_clustering(embeddings, number_of_clusters)
                # find text pieces most central in the cluster
                indices = []
                for centroid in centroids:
                    _, index = nearest_neighbor(embeddings, centroid)
                    indices.append(index)
                chunk_texts = [collection['documents'][index] for index in indices]
                chunk_ids = [collection['ids'][index] for index in indices]
                chunks = zip(chunk_ids, chunk_texts)
                for chunk in chunks:
                    print(f"MAP_REDUCE: chunk processed = {chunk[0]}, text = {chunk[1]}")

                chunks_joined = '\n\nNext Summary Piece: '.join(chunk_texts)
                print(f"chunks_joined = {chunks_joined}")

                summarize_prompt = f'''
                          Join the following text pieces and write a combined summary.
                          Be elaborate and write a comprehensible, nice summary.
                          Return only the summary, no other text. {str(chunks_joined)}
                          '''
                summary = self.llm.invoke(summarize_prompt)
            # else if Hybrid method is chosen
            elif self.summary_method == "Hybrid":
                # extract data from vector store
                embeddings = np.array(collection['embeddings'])
                # cluster (KNN)
                number_of_clusters = settings.SUMMARIZER_CENTROIDS
                _, centroids = kmeans_clustering(embeddings, number_of_clusters)
                # find text pieces most central in the cluster
                indices = []
                for centroid in centroids:
                    _, index = nearest_neighbor(embeddings, centroid)
                    indices.append(index)
                chunk_ids = [collection['ids'][index] for index in indices]
                chunk_texts = [collection['documents'][index] for index in indices]
                chunks = zip(chunk_ids, chunk_texts)
                first = True
                for chunk in chunks:
                    print(chunk[0])
                    text = chunk[1]
                    if first:
                        summarize_prompt = f'''
                                          Summarize the following text: {text}.
                                          Only return the summary, no explanation.
                                          '''
                        summary = self.llm.invoke(summarize_prompt)
                        first = False
                    else:
                        refine_prompt = f'''
                                        Given the following summary: {summary}
                                        \nRefine the summary, with the following added information: {text}
                                        \nOnly return the summary, no other text.
                                        '''
                        summary = self.llm.invoke(refine_prompt)

            # define output file and store summary
            result = os.path.join(self.content_folder, "summaries", str(file_name) + "_" +
                                  str.lower(self.summary_method) + ".txt")
            with open(file=result, mode="w", encoding="utf8") as f:
                f.write(summary.content)
