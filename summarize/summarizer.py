import os
from dotenv import load_dotenv
from loguru import logger
# local imports
import settings
# from ingest.content_iterator import ContentIterator
from ingest.ingest_utils import IngestUtils
from ingest.file_parser import FileParser
import utils as ut
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import numpy as np



def initialize_centroids(data, k):
    """ Randomly initialize centroids """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    """ Assign data points to the nearest centroid """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, clusters, k):
    """ Update centroids as the mean of assigned data points """
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
    """ K-means clustering algorithm """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids
    return clusters, centroids

def euclidean_distance(a, b):
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
    '''
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
    '''
    def __init__(self, collection_name:str, content_folder: str, vectordb_folder: str, summary_method: str,
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, local_api_url=None,
                 file_no=None, llm_type=None, llm_model_type=None, azureopenai_api_version=None):
        load_dotenv()
        self.collection_name = collection_name
        self.summary_method = summary_method
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.API_URL if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.azureopenai_api_version = settings.AZUREOPENAI_API_VERSION \
            if azureopenai_api_version is None and settings.AZUREOPENAI_API_VERSION is not None else azureopenai_api_version

        # if llm_type is "chatopenai"
        if self.llm_type == "chatopenai":
            # default llm_model_type value is "gpt-3.5-turbo"
            llm_model_type = "gpt-3.5-turbo"
            if self.llm_model_type == "gpt35_16":
                llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                llm_model_type = "gpt-4"
            self.llm = ChatOpenAI(
                client=None,
                model=llm_model_type,
                temperature=0,
            )
        # else, if llm_type is "huggingface"
        elif self.llm_type == "huggingface":
            # default value is llama-2, with maximum output length 512
            llm_model_type = "meta-llama/Llama-2-7b-chat-hf"
            max_length = 512
            if self.llm_model_type == 'GoogleFlan':
                llm_model_type = 'google/flan-t5-base'
                max_length = 512
            self.llm = HuggingFaceHub(repo_id=llm_model_type,
                                 model_kwargs={"temperature": 0.1,
                                               "max_length": max_length}
                                )
        # else, if llm_type is "local_llm"
        elif self.llm_type == "local_llm":
            logger.info("Use Local LLM")
            logger.info("Retrieving " + self.llm_model_type)
            if self.local_api_url is not None: # If API URL is defined, use it
                logger.info("Using local api url " + self.local_api_url)
                llm = Ollama(
                    model=self.llm_model_type, 
                    base_url=self.local_api_url,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            else:
                self.llm = Ollama(
                    model=self.llm_model_type,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            logger.info("Retrieved " + self.llm_model_type)

        # else, if llm_type is "azureopenai"
        elif self.llm_type == "azureopenai":
            logger.info("Use Azure OpenAI LLM")
            logger.info("Retrieving " + self.llm_model_type)
            self.llm = AzureChatOpenAI(
                azure_deployment=self.llm_model_type,
                azure_endpoint=self.local_api_url,
                api_version=self.azureopenai_api_version,
            )
            logger.info("Retrieved " + self.llm_model_type)




    def summarize(self) -> None:
        '''
            Creates instances of all parsers, iterates over all files in the folder
            When parameters are read from GUI, object is initiated with parameter settings listed
        '''
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        file_parser = FileParser()
        
        if self.summary_method == 'Refine':
            files_in_folder = os.listdir(self.content_folder)
            # loop over all the chunks to create a summary, refine it with every loop
            for file_num, file in enumerate(files_in_folder):
                file_path = os.path.join(self.content_folder, file)
                _, file_extension = os.path.splitext(file_path)
                if file_extension in [".docx", ".html", ".md", ".pdf", ".txt"]:
                    # extract raw text pages and metadata according to file type
                    raw_pages, metadata = file_parser.parse_file(file_path)
                else:
                    logger.info(f"Skipping ingestion of file {file} because it has extension {file[-4:]}")
                    continue
                documents = ingestutils.clean_text_to_docs(raw_pages, metadata)
                for text_chunk_num, text in enumerate(documents):
                    if file_num + text_chunk_num == 0:
                        summarize_text = f''' Summarize the following text: {text}. Only return the summary, no explanation.'''
                        summary = self.llm.invoke(summarize_text)
                    else:
                        refined_text = f''' Given the following summary: {summary}

Refine the summary, with the newly added information below: 
{text}
Only return the summary, no other text.
'''
                        summary = self.llm.invoke(refined_text)
                        summary = str(summary).strip("content='")[:-1] # strip the content out of the summary. Weirdly I could not figure out how to access the content object from the AIMEssage object.

            if 'summaries' in os.listdir(self.content_folder):
                with open(f'{self.content_folder}/summaries/summary_Refined.txt', 'w') as f:
                    f.write(summary)
            else:
                os.mkdir(self.content_folder + '/summaries')
                with open(f'{self.content_folder}/summaries/summary_Refined.txt', 'w') as f:
                    f.write(summary)
            

        elif self.summary_method == 'Map_Reduce':
            '''create/update vector store'''
            # get embeddings
            embeddings = ut.getEmbeddings(self.embeddings_provider, self.embeddings_model, self.local_api_url, self.azureopenai_api_version)
            # create empty list representing added files
            new_files = []

            if self.vecdb_type == "chromadb":
                # get all files in the folder
                files_in_folder = os.listdir(self.content_folder)
                # if the vector store already exists, get the set of ingested files from the vector store
                if os.path.exists(self.vectordb_folder):
                    # get chroma vector store
                    vector_store = ut.get_chroma_vector_store(self.collection_name, embeddings, self.vectordb_folder)
                    logger.info(f"Vector store already exists for specified settings and folder {self.content_folder}")
                    # determine the files that are added or deleted
                    collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                    collection_ids = [int(id) for id in collection['ids']]
                    files_in_store = [metadata['filename'] for metadata in collection['metadatas']]
                    files_in_store = list(set(files_in_store))
                    # check if files were added or removed
                    new_files = [file for file in files_in_folder if file not in files_in_store]
                    files_deleted = [file for file in files_in_store if file not in files_in_folder]
                    # delete all chunks from the vector store that belong to files removed from the folder
                    if len(files_deleted) > 0:
                        logger.info(f"Files are deleted, so vector store for {self.content_folder} needs to be updated")
                        idx_id_to_delete = []
                        for idx in range(len(collection['ids'])):
                            idx_id = collection['ids'][idx]
                            idx_metadata = collection['metadatas'][idx]
                            if idx_metadata['filename'] in files_deleted:
                                idx_id_to_delete.append(idx_id)
                        vector_store.delete(idx_id_to_delete)
                        logger.info(f"Deleted files from vectorstore")
                    # determine updated maximum id from collection after deletions
                    collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                    collection_ids = [int(id) for id in collection['ids']]
                    start_id = max(collection_ids) + 1
                # else it needs to be created first
                else:
                    logger.info(f"Vector store to be created for folder {self.content_folder}")
                    # get chroma vector store
                    vector_store = ut.get_chroma_vector_store(self.collection_name, embeddings, self.vectordb_folder)
                    collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                    # all files in the folder are to be ingested into the vector store
                    new_files = [file for file in files_in_folder]
                    start_id = 0

                # If there are any files to be ingested into the vector store
                if len(new_files) > 0:
                    logger.info(f"Files are added, so vector store for {self.content_folder} needs to be updated")
                    for file in new_files:
                        file_path = os.path.join(self.content_folder, file)
                        _, file_extension = os.path.splitext(file_path)
                        if file_extension in [".docx", ".html", ".md", ".pdf", ".txt"]:
                            # extract raw text pages and metadata according to file type
                            raw_pages, metadata = file_parser.parse_file(file_path)
                        else:
                            logger.info(f"Skipping ingestion of file {file} because it has extension {file[-4:]}")
                            continue
                        # convert the raw text to cleaned text chunks
                        documents = ingestutils.clean_text_to_docs(raw_pages, metadata)
                        logger.info(f"Extracted {len(documents)} chunks from {file}")    
                        # and add the chunks to the vector store 
                        vector_store.add_documents(
                            documents=documents,
                            embedding=embeddings,
                            collection_name=self.collection_name,
                            persist_directory=self.vectordb_folder,
                            ids=[str(id) for id in list(range(start_id, start_id + len(documents)))] # add id to file chunks for later identification
                        )
                        collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                        collection_ids = [int(id) for id in collection['ids']]
                        start_id = max(collection_ids) + 1
                    logger.info(f"Added files to vectorstore")

                # save updated vector store to disk
                vector_store.persist()
            
            # extract data from vector store
            collection = vector_store.get(include=['embeddings', 'documents'])
            embeddings = np.array(collection['embeddings'])
            # cluster (KNN)
            number_of_cluster = 3
            clusters, centroids = kmeans_clustering(embeddings, number_of_cluster)
            # find text pieces most central in the cluster
            indices = []
            for centroid in centroids:
                _, index = nearest_neighbor(embeddings, centroid)
                indices.append(index)
            text_pieces = [collection['documents'][index] for index in indices]
            text_pieces_joined = 'Next Summary Piece: '.join(text_pieces)
            prompt = f''' Join the following text pieces and write a combined summary. Be elaborate and write a comprehensible, nice summary. Return only the summary, no other text. {str(text_pieces_joined)} '''
            summary = self.llm.invoke(prompt)
            summary = str(summary).strip("content='")[:-1] # strip the content out of the summary. Weirdly I could not figure out how to access the content object from the AIMEssage object.
            # summarize
            if 'summaries' in os.listdir(self.content_folder):
                with open(f'{self.content_folder}/summaries/summary_Map_Reduce.txt', 'w') as f:
                    f.write(summary)
            else:
                os.mkdir(self.content_folder + '/summaries')
                with open(f'{self.content_folder}/summaries/summary_Map_Reduce.txt', 'w') as f:
                    f.write(summary)

                    
                
