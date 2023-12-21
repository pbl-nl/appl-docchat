import os
import streamlit as st
from PIL import Image
from loguru import logger
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import settings
import utils as ut


def click_GO_button():
    st.session_state['is_GO_clicked'] = True


def create_or_update_vectordb(content_folder_name_selected, content_folder_path_selected, vectordb_folder_path_selected):
    ingester = Ingester(content_folder_name_selected, content_folder_path_selected, vectordb_folder_path_selected)
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
    for folder_name in os.listdir(settings.DOC_DIR):
        folder_path = os.path.join(settings.DOC_DIR, folder_name)
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
    return folder_name_selected, folder_path_selected, vectordb_folder_path_selected


def check_vectordb(querier, folder_name_selected, folder_path_selected, vectordb_folder_path_selected):
    # If a folder is chosen that is not equal to the last known source folder
    if folder_name_selected != st.session_state['folder_selected']:
        # set session state of is_GO_clicked to False (will be set to True when OK button is clicked)
        st.session_state['is_GO_clicked'] = False
        # clear all chat messages on screen and in Querier object
        st.session_state['messages'] = []
        querier.clear_history()
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given in settings.py, create it first
    if not os.path.exists(vectordb_folder_path_selected):
        logger.info("Creating vectordb")
        spinner_message = f'Creating vector database for folder {folder_name_selected}. Depending on the size, this may take a while. Please wait...'
    else:
        logger.info("Updating vectordb")
        spinner_message = f'Checking if vector database needs an update for folder {folder_name_selected}. This may take a while, please wait...'
    with st.spinner(spinner_message):
        create_or_update_vectordb(folder_name_selected, folder_path_selected, vectordb_folder_path_selected)

    # create a new chain based on the new source folder 
    querier.make_chain(folder_name_selected, vectordb_folder_path_selected)
    # set session state of selected folder to new source folder 
    st.session_state['folder_selected'] = folder_name_selected
    logger.info("Executed check_vectordb")


def handle_query(querier, prompt: str):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.spinner(f'Thinking...'):
        # Generate a response
        response, scores = querier.ask_question(prompt)
    # Display the response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    # Add the response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response["answer"]})
    if len(response["source_documents"]) > 0:
        with st.expander("Sources used for answer"):
            cnt = 0
            for document in response["source_documents"]:
                st.markdown(f"**page: {document.metadata['page_number']}, chunk: {document.metadata['chunk']}, score: {scores[cnt]:.4f}, file: {document.metadata['filename']}**")
                cnt += 1
                st.markdown(f"{document.page_content}")
    else:
        logger.info("No source documents found relating to the question")
    logger.info("Executed handle_query(querier, prompt)")


@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    Also prepares the sidebar with folder list
    """
    imagecol, headercol = st.columns([0.3, 0.7])
    logo_image = Image.open(settings.APP_LOGO)
    with imagecol:
        st.image(logo_image, width=250)
    with headercol:
        st.header(settings.APP_HEADER)
    # set session state default for messages to fight hallucinations
    # st.session_state.setdefault('messages', [{"role": "system", "content": "You are a helpful assistant. 
    # Custom CSS to have white expander background
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: white;
            color: black; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: white;
            color: black; # Expander content color
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    with st.sidebar.expander("User manual"):
        # read app explanation from file explanation.txt
        with open(file=settings.APP_INFO) as file:
            explanation = file.read()
        st.markdown(body=explanation, unsafe_allow_html=True)
        st.image("./images/multilingual.png")
    st.sidebar.divider()
    # Sidebar text for folder selection
    st.sidebar.title("Select a document folder")
    logger.info("Executed initialize_page()")


def initialize_session_state():
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


@st.cache_resource
def initialize_querier():
    """
    Create a Querier object
    """
    querier = Querier()
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
folder_name_selected, folder_path_selected, vectordb_folder_path_selected = folder_selector(source_folders_available)

# Create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
st.sidebar.button("GO", type="primary", on_click=click_GO_button)

# Only start a conversation when a folder is selected and selection is confirmed with "GO" button
if st.session_state['is_GO_clicked']:
    # create or update vector database if necessary
    check_vectordb(querier, folder_name_selected, folder_path_selected, vectordb_folder_path_selected)
    # If a summary is required
    chk_summary = st.sidebar.checkbox(label="show summary")
    if chk_summary:
        with st.sidebar.expander("Summary"):
            st.markdown(body="summary comes here", unsafe_allow_html=True)

    # Show button "Clear Conversation"
    clear_messages_button = st.button("Clear Conversation", key="clear")
    
    # If button "Clear Conversation" is clicked
    if clear_messages_button:
        # clear all chat messages on screen and in Querier object
        # NB: session state of "is_GO_clicked" and "folder_selected" remain unchanged
        st.session_state['messages'] = []
        querier.clear_history()
        logger.info("Clear Conversation button clicked")

    # Display chat messages from history 
    display_chat_history()

    # React to user input if a question has been asked
    if prompt := st.chat_input("Your question"):
        handle_query(querier, prompt)
