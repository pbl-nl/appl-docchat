import os

def create_vectordb_name(content_folder_name, doc_dir, vecdb_dir, vecdb_type, embeddings_type, chunk_size, chunk_overlap):
    content_folder_path = os.path.join(doc_dir, content_folder_name)
    # vectordb_name is created from vecdb_type, chunk_size, chunk_overlap, embeddings_type 
    vectordb_name = "_" + vecdb_type + "_" + str(chunk_size) + "_" + str(chunk_overlap) + "_" + embeddings_type
    vectordb_folder_path = os.path.join(vecdb_dir, content_folder_name) + vectordb_name 
    return content_folder_path, vectordb_folder_path
