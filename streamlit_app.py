import os
import streamlit as st
# from PIL import Image
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
from settings import APP_INFO, APP_HEADER, DOC_DIR, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE
import utils


def click_folder_selected_button():
    st.session_state['folder_selected'] = True


def display_chat_history():
    for message in st.session_state['messages']:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def folderlist_creator(folder_path):
    folders = []
    for file in os.listdir(folder_path):
        folder = os.path.join(folder_path, file)
        if os.path.isdir(folder):
            folders.append(file)
    return folders


def folder_selector(folders):
    # Get source folder with docs from user
    content_folder_name = st.sidebar.selectbox("label=folder_selector", options=folders, on_change=unclick_folder_selected_button, label_visibility="hidden")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    return content_folder_name, content_folder_path, vectordb_folder_path


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
    # Let user choose a folder with docs in the sidebar
    st.sidebar.title("Select a document folder")
    # necessary at first start of session
    if "folder_selected" not in st.session_state:
        st.session_state['folder_selected'] = False
    if "messages" not in st.session_state:
        st.session_state['messages'] = []


@st.cache_resource
def initialize_querier(input_folder: str, vectordb_folder: str):
    """
    Create a Querier object
    """
    querier = Querier(input_folder, vectordb_folder, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    print("Querier object created")
    return querier


def set_page_config():
    st.set_page_config(page_title="Chat with your documents", page_icon=':books:', layout='wide', initial_sidebar_state='auto')


def unclick_folder_selected_button():
    st.session_state['folder_selected'] = False
    st.session_state['messages'] = []


#### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()
# initialize page. Needs to be done before any action from the user
initialize_page()
# return list of folder names
source_folders_available = folderlist_creator(DOC_DIR)

# Initialise session state variables
if 'folder_selected' not in st.session_state:
    st.session_state['folder_selected'] = False
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Chosen folder and associated vector database
content_folder_name_selected, content_folder_path_selected, vectordb_folder_selected = folder_selector(source_folders_available)

# Button to confirm folder selction
st.sidebar.button("OK", on_click=click_folder_selected_button)

# Start a conversation when a folder is selected
if st.session_state['folder_selected']:
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given in settings.py, create it first
    if not os.path.exists(vectordb_folder_selected):
        with st.spinner(f'Creating vector database for folder {content_folder_name_selected}. Depending on the size of the source folder, this may take a while. Please wait...'):
            ingester = Ingester(content_folder_name_selected, content_folder_path_selected, vectordb_folder_selected, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
            ingester.ingest()
    
    # Creation of Querier object
    querier = initialize_querier(content_folder_name_selected, vectordb_folder_selected)
    
    # Show button "Clear Conversation"
    clear_messages_button = st.button("Clear Conversation", key="clear")
    # If button "Clear Conversation" is clicked, remove all existing messages, if any
    if clear_messages_button:
        st.session_state['messages'] = []

    # Display chat messages from history on app rerun
    display_chat_history()

    # React to user input if a question has been asked
    if prompt := st.chat_input("Ask your question"):
        handle_query(querier, prompt)
