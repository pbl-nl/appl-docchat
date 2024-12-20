import os
import streamlit as st
from PIL import Image
from loguru import logger
from dotenv import load_dotenv
from typing import List
# local imports
import settings
import pandas as pd
import utils as ut
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator


def click_go_button():
    st.session_state['is_GO_clicked'] = True


def click_exit_button():
    st.session_state['is_EXIT_clicked'] = True


def get_chunks(my_embeddings_model: str, folder_name_selected: str, vectorstore_folder_path: str, prompt: str):
    # get embeddings
    my_embeddings = EmbeddingsCreator(embeddings_provider = "azureopenai",
                                      embeddings_model = my_embeddings_model).get_embeddings()

    # get vector store
    vector_store = VectorStoreCreator(settings.VECDB_TYPE).get_vectorstore(embeddings=my_embeddings,
                                                                           content_folder=folder_name_selected,
                                                                           vecdb_folder=vectorstore_folder_path)
    # determine the files that are added or deleted
    collection = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
    collection_size = len(collection['ids'])
    chunks = []
    for idx in range(collection_size):
        idx_metadata = collection['metadatas'][idx]
        filename = idx_metadata["filename"]
        page = idx_metadata["page_number"]
        chunk = idx_metadata["chunk"]
        idx_document = collection['documents'][idx]
        similarity = None
        # if user enters a prompt
        if prompt != "":
            # convert chunk to numerical vector (! cannot use collection['embeddings'][idx] ??)
            chunk_vector = my_embeddings.embed_documents([idx_document])[0]
            # convert user prompt to numerical vector
            prompt_vector = my_embeddings.embed_documents([prompt])[0]
            # calculate all similarities between vector store chunks and prompt
            similarity = ut.cosine_similarity(a=prompt_vector,
                                              b=chunk_vector)
        chunks.append((filename, page, chunk, idx_document, similarity))

    if prompt != "":
        chunks = sorted(chunks, key=lambda x: (-x[4], x[0], x[1], x[2]))
    else:
        chunks = sorted(chunks, key=lambda x: (x[0], x[1], x[2]))

    return collection_size, chunks

@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    """
    st.header("Chunks analysis")
    logo_image = Image.open(settings.APP_LOGO)
    st.sidebar.image(logo_image, width=250)
    load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
    logger.info("Executed initialize_page()")


def initialize_session_state():
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'is_EXIT_clicked' not in st.session_state:
        st.session_state['is_EXIT_clicked'] = False


def set_page_config():
    st.set_page_config(page_title="Chunks analysis", page_icon=':books:', layout='wide')
    logger.info("\nExecuted set_page_config()")


# ### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()
# Initialize page, executed only once per session
initialize_page()
# initialize session state variables
initialize_session_state()
# Create button to exit the application. This button sets session_state['is_EXIT_clicked'] to True
st.sidebar.button("EXIT", type="primary", on_click=click_exit_button)
# allow user to set the path to the document folder
folder_path_selected = st.sidebar.text_input(label="***ENTER THE DOCUMENT FOLDER PATH***",
                                             help="""Please enter the full path e.g. Y:/User/troosts/chatpbl/...""")
user_query = st.sidebar.text_input(label="***ENTER YOUR PROMPT***",
                                   help="""Enter prompt as used in review.py""")
if st.session_state['is_EXIT_clicked']:
    ut.exit_UI()

if folder_path_selected != "":
    load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
    # define the selected folder name
    folder_name_selected = os.path.basename(folder_path_selected)
    # get the names of the vector database folders
    vectorstore_folders = os.listdir(os.path.join(folder_path_selected, "vector_stores"))
    num_vectorstores = len(vectorstore_folders)
    columns = st.columns(num_vectorstores)

    # Create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
    st.sidebar.button("GO", type="primary", on_click=click_go_button)

    if st.session_state['is_GO_clicked']:
        first_vectorstore = True
        # For each folder
        for i, col in enumerate(columns):
            vectorstore_folder  = vectorstore_folders[i]
            vectorstore_folder_path = os.path.join(folder_path_selected, "vector_stores", vectorstore_folder)
            #  extract the settings that were used to create the folder
            vectorstore_settings = vectorstore_folder.split("_")
            retriever_type = vectorstore_settings[0]
            embedding_model = vectorstore_settings[1]
            text_splitter_method = vectorstore_settings[2]
            if retriever_type == "parent":
                chunk_k = vectorstore_settings[5]
                chunk_overlap = vectorstore_settings[6]
            else:
                chunk_k = vectorstore_settings[3]
                chunk_overlap = vectorstore_settings[4]
            
            # Store settings in a dataframe
            df = pd.DataFrame(columns=[vectorstore_folder])
            df.loc[len(df)] = f"retriever_type = {retriever_type}"
            df.loc[len(df)] = f"embedding_model = {embedding_model}"
            df.loc[len(df)] = f"text_splitter = {text_splitter_method}"
            df.loc[len(df)] = f"chunk_k = {chunk_k}"
            df.loc[len(df)] = f"chunk_overlap = {chunk_overlap}"
            # get chunk info from vectorstore
            collection_size, chunks = get_chunks(my_embeddings_model=embedding_model,
                                                 folder_name_selected=folder_name_selected,
                                                 vectorstore_folder_path=vectorstore_folder_path,
                                                 prompt=user_query)
            df.loc[len(df)] = f"number of chunks: {collection_size}"
            # show settings
            with col:
                st.dataframe(data=df, use_container_width=True, hide_index=True)
            
            # show vector store contents
            df_cont = pd.DataFrame(columns=["filename", "page", "chunk", "text", "similarity"])
            for chunk in chunks:
                df_cont.loc[len(df_cont)] = [chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]]
            with col:
                st.dataframe(data=df_cont, height=600, use_container_width=True, hide_index=True)
