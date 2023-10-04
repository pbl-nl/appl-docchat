import os
from loguru import logger
# local imports
from ingest.ingester import Ingester
import settings
import utils

def main():
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name, 
                                                                           settings.DOC_DIR, 
                                                                           settings.VECDB_DIR, 
                                                                           settings.VECDB_TYPE, 
                                                                           settings.EMBEDDINGS_TYPE, 
                                                                           settings.CHUNK_SIZE, 
                                                                           settings.CHUNK_OVERLAP)
    if not os.path.exists(vectordb_folder_path):
        ingester = Ingester(content_folder_name, 
                            content_folder_path, 
                            vectordb_folder_path, 
                            settings.EMBEDDINGS_TYPE, 
                            settings.VECDB_TYPE, 
                            settings.CHUNK_SIZE, 
                            settings.CHUNK_OVERLAP)
        ingester.ingest()
        logger.info(f"Created Chroma vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Chroma vector store already exists for folder {content_folder_name}")


if __name__ == "__main__":
    main()