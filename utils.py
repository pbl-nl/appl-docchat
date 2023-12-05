import os
import datetime as dt
from loguru import logger
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
# local imports
import settings


def create_vectordb_name(content_folder_name, chunk_size=None, chunk_overlap=None):
    content_folder_path = os.path.join(settings.DOC_DIR, content_folder_name)
    # vectordb_name is created from vecdb_type, chunk_size, chunk_overlap, embeddings_type 
    if chunk_size:
        vectordb_name = "_" + settings.VECDB_TYPE + "_" + str(chunk_size) + "_" + str(chunk_overlap) + "_" + settings.EMBEDDINGS_PROVIDER
    else:
        vectordb_name = "_" + settings.VECDB_TYPE + "_" + str(settings.CHUNK_SIZE) + "_" + str(settings.CHUNK_OVERLAP) + "_" + settings.EMBEDDINGS_PROVIDER
    vectordb_folder_path = os.path.join(settings.VECDB_DIR, content_folder_name) + vectordb_name 
    return content_folder_path, vectordb_folder_path


def get_chroma_vector_store(collection_name, embeddings, vectordb_folder):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=vectordb_folder,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store


def get_settings_as_dictionary(file_name):
    # Initialize an empty dictionary to store the variables and their values
    variables_dict = {}
    # Open and read the file
    with open(file_name, 'r') as file:
        lines = file.readlines()
    start_reading = False
    # Process each line in the file
    for line in lines:
        # start reading below the line with ####
        if line.startswith("####"):
            start_reading = True
        # ignore comment lines
        if start_reading and not line.startswith("#"):
            # Remove leading and trailing whitespace and split the line by '='
            parts = line.strip().split('=')
            # Check if the line can be split into two parts
            if len(parts) == 2:
                # Extract the variable name and value
                variable_name = parts[0].strip()
                variable_value = parts[1].strip()
                # Use exec() to assign the value to the variable name
                exec(f'{variable_name} = {variable_value}')
                # Add the variable and its value to the dictionary
                variables_dict[variable_name] = eval(variable_name)
    return variables_dict


def getEmbeddings(embeddings_provider, embeddings_model, local_api_url):
    # determine embeddings model
    if embeddings_provider == "openai":
        embeddings = OpenAIEmbeddings(model=embeddings_model, client=None)
        logger.info("Loaded openai embeddings")
    elif embeddings_provider == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    elif embeddings_provider == "local_embeddings":
        if local_api_url is not None:
            embeddings = OllamaEmbeddings(
                base_url = local_api_url,
                model = embeddings_model)
        else:
            embeddings = OllamaEmbeddings(
                model = embeddings_model)
        logger.info("Loaded local embeddings: " + embeddings_model)
    return embeddings


def get_timestamp():
    return str(dt.datetime.now())




