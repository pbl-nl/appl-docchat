import os
import streamlit as st
# from PIL import Image
from loguru import logger
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
from settings import APP_INFO, APP_HEADER, DOC_DIR, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE
import utils


def create_vectordb(content_folder_name_selected, content_folder_path_selected, vectordb_folder_path_selected):
    ingester = Ingester(content_folder_name_selected, content_folder_path_selected, vectordb_folder_path_selected, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    ingester.ingest()



def display_chat_history():
    for message in st.session_state['messages']:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    logger.info("Executed display_chat_history()")


def folderlist_creator():
    """
    Creates a list of folder names (without path).
    Folder names are found in DOC_DIR (see settings).
    """
    folders = []
    for folder_name in os.listdir(DOC_DIR):
        folder_path = os.path.join(DOC_DIR, folder_name)
        if os.path.isdir(folder_path):
            folders.append(folder_name)
    logger.info("Executed folderlist_creator()")
    return folders


def folder_selector(querier, folders):
    # Select source folder with docs
    folder_name_selected = st.sidebar.selectbox("label=folder_selector", options=folders, label_visibility="hidden")
    logger.info(f"folder_name_selected is now {folder_name_selected}")
    # get associated source folder path and vectordb path
    folder_path_selected, vectordb_folder_path_selected = utils.create_vectordb_name(folder_name_selected)
    logger.info(f"vectordb_folder_path_selected is now {vectordb_folder_path_selected}")
    # If a folder is chosen that is not equal to the last know source folder
    if folder_name_selected != st.session_state['folder_selected']:
        # set session state of is_folder_selected to False (will be set to True when OK button is clicked)
        st.session_state['is_folder_selected'] = False
        # clear all chat messages on screen and in Querier object
        st.session_state['messages'] = []
        querier.clear_history()
        # When the associated vector database of the chosen content folder doesn't exist with the settings as given in settings.py, create it first
        if not os.path.exists(vectordb_folder_path_selected):
            logger.info("Creating vectordb")
            with st.spinner(f'Creating vector database for folder {folder_name_selected}. Depending on the size of the source folder, this may take a while. Please wait...'):
                create_vectordb(folder_name_selected, folder_path_selected, vectordb_folder_path_selected)
        # create a new chain based on the new source folder 
        querier.make_chain(folder_name_selected, vectordb_folder_path_selected)
        # set session state of selected folder to new source folder 
        st.session_state['folder_selected'] = folder_name_selected
        logger.info(f"Content folder name is now {folder_name_selected}")
    logger.info("Executed folder_selector(folders)")
    return folder_name_selected, folder_path_selected, vectordb_folder_path_selected


def handle_query(querier, prompt: str):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # Generate a response
    response, sources = querier.ask_question(prompt)
    # Display the response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add the response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.expander("Show sources"):
        st.write(sources)
    logger.info("Executed handle_query(querier, prompt)")


@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    Also prepares the sidebar with folder list
    """
    imagecol, headercol = st.columns([0.3, 0.7])
    # logo_image = Image.open(APP_LOGO)
    # with imagecol:
    #     st.image(logo_image, width=250)
    with headercol:
        st.header(APP_HEADER)
    # set session state default for messages
    # st.session_state.setdefault('messages', [{"role": "system", "content": "You are a helpful assistant. If the answer to the question cannot be found in the context, just answer that you don't know the answer because the given context doesn't provide information"}])
    with st.expander("Show explanation how to use this application"):
        # read app explanation from file explanation.txt
        with open(file=APP_INFO) as file:
            explanation = file.read()
        st.markdown(body=explanation, unsafe_allow_html=True)
        st.image("./images/multilingual.png")
    st.divider()
    # Sidebar text for folder selection
    st.sidebar.title("Select a document folder")
    logger.info("Executed initialize_page()")


def initialize_session_state():
    if 'is_folder_selected' not in st.session_state:
        st.session_state['is_folder_selected'] = False
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


@st.cache_resource
def initialize_querier():
    """
    Create a Querier object
    """
    querier = Querier(EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info("Executed initialize_querier()")
    return querier


def set_page_config():
    st.set_page_config(page_title="Chat with your documents", page_icon=':books:', layout='wide', initial_sidebar_state='auto')
    logger.info("\nExecuted set_page_config()")




#### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()
# Initialize page, executed only once per session
initialize_page()
# create list of content folders
source_folders_available = folderlist_creator()
# initialize session state variables
initialize_session_state()
# Creation of Querier object, executed only once per session
querier = initialize_querier()

# Chosen folder and associated vector database
content_folder_name_selected, content_folder_path_selected, vectordb_folder_path_selected = folder_selector(querier, source_folders_available)

# Create button to confirm folder selection. This button sets session_state['is_folder_selected'] to True
is_folder_selected_button = st.sidebar.button("OK", type="primary")
if is_folder_selected_button:
    st.session_state['is_folder_selected'] = True

# Only start a conversation when a folder is selected and selection is confirmed with "OK" button
if st.session_state['is_folder_selected']:
    # Show button "Clear Conversation"
    clear_messages_button = st.button("Clear Conversation", key="clear")
    # If button "Clear Conversation" is clicked
    if clear_messages_button:
        # clear all chat messages on screen and in Querier object
        # NB: session state of is_folder_selected, folder_selected remain unchanged
        st.session_state['messages'] = []
        querier.clear_history()
        logger.info("Clear Conversation button clicked")

    # Display chat messages from history on app rerun
    display_chat_history()

    # React to user input if a question has been asked
    if prompt := st.chat_input("Ask your question"):
        with st.spinner(f'Thinking...'):
            handle_query(querier, prompt)
