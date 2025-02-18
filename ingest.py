"""
This module can be run in stand alone fashion using 'python ingest.py'.
It ingests documents when necessary from a given folder into a persistent vector database
"""
import os
from loguru import logger
# local imports
from ingest.ingester import Ingester
import utils as ut


def main():
    """
    Ingests documents when necessary from a given folder into a persistent vector database
    """
    # Get source folder with docs from user
    content_folder_path = input("Source folder path of documents (including path): ")
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    # Get private docs indicator from user
    # confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    # confidential = confidential_yn in ["y", "Y"]
    confidential = False
    # get relevant models
    _, _, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                         private=confidential)
    # get associated source folder path and vectordb path
    vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                embeddings_provider=embeddings_provider,
                                                embeddings_model=embeddings_model)
    # create subfolder for storage of vector databases if not existing
    ut.create_vectordb_folder(content_folder_path)
    # store documents in vector database if necessary
    ingester = Ingester(collection_name=content_folder_name,
                        content_folder=content_folder_path,
                        vecdb_folder=vecdb_folder_path,
                        embeddings_provider=embeddings_provider,
                        embeddings_model=embeddings_model)

    ingester.ingest()
    logger.info(f"finished ingesting documents for folder {content_folder_path}")


if __name__ == "__main__":
    main()
