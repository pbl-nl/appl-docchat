import os
from settings import DOC_DIR, VECDB_DIR, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE

def create_vectordb_name(content_folder_name):
    content_folder_path = os.path.join(DOC_DIR, content_folder_name)
    vectordb_name = "_" + VECDB_TYPE + "_" + str(CHUNK_SIZE) + "_" + str(CHUNK_OVERLAP) + "_" + EMBEDDINGS_TYPE
    vectordb_folder_path = os.path.join(VECDB_DIR, content_folder_name) + vectordb_name 
    return content_folder_path, vectordb_folder_path
