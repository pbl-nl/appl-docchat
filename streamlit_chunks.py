import os
import streamlit as st
from PIL import Image
from loguru import logger
from dotenv import load_dotenv
# local imports
import settings
import pandas as pd
import utils as ut
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator


def click_GO_button():
    st.session_state['is_GO_clicked'] = True


def get_chunks(my_embeddings_model: str, folder_name_selected: str, vectorstore_folder_path: str):
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
        chunks.append((filename, page, chunk, f"page: {page}, chunk: {chunk}\n\n" + idx_document))

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
    load_dotenv()
    logger.info("Executed initialize_page()")


def initialize_session_state():
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False


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
# allow user to set the path to the document folder
folder_path_selected = st.sidebar.text_input(label="***ENTER THE DOCUMENT FOLDER PATH***",
                                             help="""Please enter the full path e.g. Y:/User/troosts/chatpbl/...""")
if folder_path_selected != "":
    load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
    # define the selected folder name
    folder_name_selected = os.path.basename(folder_path_selected)
    # get the names of the vector database folders
    vectorstore_folders = os.listdir(os.path.join(folder_path_selected, "vector_stores"))
    num_vectorstores = len(vectorstore_folders)
    columns = st.columns(num_vectorstores)

    # Create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
    st.sidebar.button("GO", type="primary", on_click=click_GO_button)

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
            # define a dataframe and add settings
            df = pd.DataFrame(columns=[vectorstore_folder])
            df.loc[len(df)] = f"retriever_type = {retriever_type}"
            df.loc[len(df)] = f"embedding_model = {embedding_model}"
            df.loc[len(df)] = f"text_splitter = {text_splitter_method}"
            df.loc[len(df)] = f"chunk_k = {chunk_k}"
            df.loc[len(df)] = f"chunk_overlap = {chunk_overlap}"
            # add chunks in vectorstore to dataframe
            collection_size, chunks = get_chunks(my_embeddings_model=embedding_model,
                                                 folder_name_selected=folder_name_selected,
                                                 vectorstore_folder_path=vectorstore_folder_path)
            df.loc[len(df)] = ""
            df.loc[len(df)] = f"number of chunks: {collection_size}"
            prv_file = ""
            file_num = 0
            for chunk in chunks:
                if chunk[0] != prv_file:
                    if file_num > 0:
                        df.loc[len(df)] = ""
                    prv_file = chunk[0]
                    file_num += 1
                    df.loc[len(df)] = f"filename: {chunk[0]}"
                else:
                    df.loc[len(df)] = chunk[3]
            with col:
                st.dataframe(data=df, height=1000, use_container_width=True, hide_index=True)
