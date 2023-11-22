from loguru import logger
# local imports
from ingest.ingester import Ingester
import utils

def main():
    '''
        Creates an instance of Ingester class and ingests documents when necessary
    '''
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    ingester = Ingester(collection_name=content_folder_name, content_folder=content_folder_path, vectordb_folder=vectordb_folder_path)
    ingester.ingest()
    logger.info(f"finished ingesting documents for folder {content_folder_name}")
     
if __name__ == "__main__":
    main()