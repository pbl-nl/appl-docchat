from loguru import logger
# local imports
from ingest.ingester import Ingester
import utils


def main():
    """
        Ingests documents when necessary from a given folder into a persistent vector database
    """
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vecdb_folder_path = utils.create_vectordb_name(content_folder_name)
    ingester = Ingester(collection_name=content_folder_name,
                        content_folder=content_folder_path,
                        vecdb_folder=vecdb_folder_path)
    ingester.ingest()
    logger.info(f"finished ingesting documents for folder {content_folder_name}")


if __name__ == "__main__":
    main()
