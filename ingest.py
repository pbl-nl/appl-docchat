import os
# local imports
from ingest.ingester import Ingester
from settings import DOC_DIR, VECDB_DIR, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE

def main():
    # Get source folder with docs from user
    input_folder = input("Source folder of documents (without path): ")
    selected_folder = os.path.join(DOC_DIR, input_folder)
    vectordb_name = "_" + VECDB_TYPE + "_" + str(CHUNK_SIZE) + "_" + str(CHUNK_OVERLAP) + "_" + EMBEDDINGS_TYPE
    vectordb_folder = os.path.join(VECDB_DIR, input_folder) + vectordb_name 

    ingester = Ingester(input_folder, selected_folder, vectordb_folder, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    ingester.ingest()


if __name__ == "__main__":
    main()