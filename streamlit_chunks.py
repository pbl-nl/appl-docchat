import os
import streamlit as st
from PIL import Image
from loguru import logger
from dotenv import load_dotenv
# local imports
import settings
import pandas as pd
import utils as ut


def click_GO_button():
    st.session_state['is_GO_clicked'] = True


def folderlist_creator():
    """
    Creates a list of folder names (without path).
    Folder names are found in DOC_DIR (see settings).
    """
    folders = []
    for folder_name in os.listdir(settings.CHUNK_DIR):
        folder_path = os.path.join(settings.CHUNK_DIR, folder_name)
        if os.path.isdir(folder_path):
            folders.append(folder_name)
    logger.info("Executed folderlist_creator()")
    return folders


def folder_selector(folders):
    # Select source folder with docs
    folder_name_selected = st.sidebar.selectbox("label=folder_selector", options=folders, label_visibility="hidden")
    logger.info(f"folder_name_selected is now {folder_name_selected}")
    # get associated source folder path and vectordb path
    folder_path_selected, vectordb_folder_path_selected = ut.create_vectordb_name(folder_name_selected)
    logger.info(f"vectordb_folder_path_selected is now {vectordb_folder_path_selected}")
    if folder_name_selected != st.session_state['folder_selected']:
        st.session_state['is_GO_clicked'] = False
    # set session state of selected folder to new source folder 
    st.session_state['folder_selected'] = folder_name_selected
    return folder_name_selected, folder_path_selected


def get_chunks(my_folder):
    # get collection names available in vectordb
    # loop over files in folder
    files_in_folder = os.listdir(my_folder)
    first = True
    for file in files_in_folder:
        if file != "all_chunks.tsv":
            path = os.path.join(my_folder, file)
            if first:
                # simply read the dataframe
                df_out = pd.read_csv(filepath_or_buffer=path, sep="\t")
                first = False
            else:
                # read the file in a dataframe and add the information to the current one
                df_in = pd.read_csv(filepath_or_buffer=path, sep="\t")
                df_out = df_out.merge(
                    df_in,
                    on=["chunk"],
                    how="outer",
                )
    path_out = os.path.join(my_folder, "all_chunks.tsv")
    df_out.to_csv(path_out, sep="\t", index=False)
    return df_out


@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    """
    imagecol, headercol = st.columns([0.3, 0.7])
    logo_image = Image.open(settings.APP_LOGO)
    with imagecol:
        st.image(logo_image, width=250)
    with headercol:
        st.header("chatNMDC: chunks")
    load_dotenv()
    logger.info("Executed initialize_page()")


def initialize_session_state():
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""


def set_page_config():
    st.set_page_config(page_title="ChatNMDC chunks", page_icon=':books:', layout='wide')
    logger.info("\nExecuted set_page_config()")


# ### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()
# Initialize page, executed only once per session
initialize_page()
# create list of vector store folders
source_folders_available = folderlist_creator()
# initialize session state variables
initialize_session_state()

# Chosen folder and associated vector database
folder_name_selected, folder_path_selected = folder_selector(source_folders_available)

# Create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
st.sidebar.button("GO", type="primary", on_click=click_GO_button)

# Only start a conversation when a folder is selected and selection is confirmed with "GO" button
if st.session_state['is_GO_clicked']:
    chunks_folder = os.path.join(settings.CHUNK_DIR, folder_name_selected)
    df_chunks = get_chunks(chunks_folder)
    st.subheader("Chunks available")
    st.dataframe(data=df_chunks, hide_index=True)
