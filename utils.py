import os
# local imports
import settings

def create_vectordb_name(content_folder_name):
    content_folder_path = os.path.join(settings.DOC_DIR, content_folder_name)
    # vectordb_name is created from vecdb_type, chunk_size, chunk_overlap, embeddings_type 
    vectordb_name = "_" + settings.VECDB_TYPE + "_" + str(settings.CHUNK_SIZE) + "_" + str(settings.CHUNK_OVERLAP) + "_" + settings.EMBEDDINGS_PROVIDER
    vectordb_folder_path = os.path.join(settings.VECDB_DIR, content_folder_name) + vectordb_name 
    return content_folder_path, vectordb_folder_path
