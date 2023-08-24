import os
# local imports
from ingest.ingester import Ingester
from settings import VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE
import utils

def main():
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    ingester = Ingester(content_folder_name, content_folder_path, vectordb_folder_path, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    ingester.ingest()


if __name__ == "__main__":
    main()