from typing import List, Tuple
import fitz
import os
import streamlit as st
from PIL import Image
from loguru import logger
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
from summarize.summarizer import Summarizer
import settings
import utils as ut


def click_go_button() -> None:
    """
    Sets session state of GO button clicked to True
    """
    st.session_state['is_GO_clicked'] = True


@st.cache_data
def create_and_show_summary(my_summary_type: str,
                            my_folder_path_selected: str,
                            my_folder_name_selected: str,
                            my_vecdb_folder_path_selected: str,
                            selected_documents: List[str]) -> None:
    """
    Creates or loads a summary of the chosen document(s) and shows it in the UI

    Parameters
    ----------
    my_summary_type : str
        chosen summary type, either "Short", "Long" or "Hybrid"
    my_folder_path_selected : str
        path of content folder
    my_folder_name_selected : str
        name of content folder
    my_vecdb_folder_path_selected : str
        path of vector database foler
    selected_documents : List[str]
        list of selected documents
    """
    if my_summary_type == "Short":
        summarization_method = "Map_Reduce"
    elif my_summary_type == "Long":
        summarization_method = "Refine"
    elif my_summary_type == "Hybrid":
        summarization_method = "Hybrid"

    # for each selected file in content folder
    with st.expander(label=f"{my_summary_type} summary",
                     expanded=True):
        first_summary = True
        files_in_folder = ut.get_relevant_files_in_folder(my_folder_path_selected)
        if selected_documents == ["All"]:
            selected_documents = files_in_folder
        for file in files_in_folder:
            if file in selected_documents:
                file_name, _ = os.path.splitext(file)
                summary_name = os.path.join(my_folder_path_selected, "summaries",
                                            str(file_name) + "_" + str.lower(summarization_method) + ".txt")
                # if summary does not exist yet, create it
                if not os.path.isfile(summary_name):
                    my_spinner_message = f'''Creating summary for {file}.\n
                                        Depending on the size of the file and the type of summary, this may take a while. Please wait...'''
                    with st.spinner(my_spinner_message):
                        summarizer = Summarizer(content_folder=my_folder_path_selected,
                                                collection_name=my_folder_name_selected,
                                                summary_method=summarization_method,
                                                vecdb_folder=my_vecdb_folder_path_selected)
                        summarizer.summarize()
                # show summary
                if not first_summary:
                    st.divider()
                with open(file=summary_name, mode="r", encoding="utf8") as f:
                    st.write(f"**{file}:**\n")
                    st.write(f.read())
                first_summary = False
    logger.info("Executed create_and_show_summary()")


def display_chat_history() -> None:
    """
    Shows the complete chat history
    """
    for message in st.session_state['messages']:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    logger.info("Executed display_chat_history()")


@st.cache_data
def folderlist_creator() -> List[str]:
    """
    Creates a list of folder names (without path).
    Folder names are found in DOC_DIR (see settings.py).

    Returns
    -------
    List[str]
        list of folder names to be shown in the UI
    """
    folders = []
    for folder_name in os.listdir(settings.DOC_DIR):
        folder_path = os.path.join(settings.DOC_DIR, folder_name)
        if os.path.isdir(folder_path):
            folders.append(folder_name)
    logger.info("Executed folderlist_creator()")

    return folders


@st.cache_data
def documentlist_creator(my_folder_path_selected: str) -> List[str]:
    """
    Creates a list of document names (without path), available in the selected folder path

    Parameters
    ----------
    folder : str
        the selected document folder

    Returns
    -------
    List[str]
        list of available and valid document names
    """
    documents = ['All']
    documents.extend(ut.get_relevant_files_in_folder(my_folder_path_selected))
    logger.info("Executed documentlist_creator()")

    return documents


def folder_selector(folders: List[str]) -> Tuple[str, str, str]:
    """
    selects a document folder and creates the asscciated document folder path and vector database path

    Parameters
    ----------
    folders : List[str]
        list of available folders

    Returns
    -------
    Tuple[str, str, str]
        tuple of selected folder name, its path and its vector database
    """
    # Select source folder with docs
    my_folder_name_selected = st.sidebar.selectbox(label="***SELECT A FOLDER***",
                                                   options=folders)
    logger.info(f"folder_name_selected is now {my_folder_name_selected}")
    # get associated source folder path and vectordb path
    my_folder_path_selected, my_vecdb_folder_path_selected = ut.create_vectordb_name(my_folder_name_selected)
    logger.info(f"vectordb_folder_path_selected is now {my_vecdb_folder_path_selected}")
    if my_folder_name_selected != st.session_state['folder_selected']:
        st.session_state['is_GO_clicked'] = False
    # set session state of selected folder to new source folder
    st.session_state['folder_selected'] = my_folder_name_selected
    logger.info("Executed folder_selector()")

    return my_folder_name_selected, my_folder_path_selected, my_vecdb_folder_path_selected


def document_selector(documents: List[str]) -> List[str]:
    """
    selects one or more valid documents from document folder

    Parameters
    ----------
    documents : List[str]
        list of available valid documents

    Returns
    -------
    List[str]
        list of selected valid documents
    """
    # Creating a multi-select dropdown
    my_document_name_selected = st.sidebar.multiselect(label="***SELECT ANY/ALL FILES***",
                                                       options=documents,
                                                       default=documents[0],
                                                       key='document_selector')
    logger.info(f"document_name_selected is now {my_document_name_selected}")
    if my_document_name_selected != st.session_state['document_selected']:
        st.session_state['is_GO_clicked'] = False
    # set session state of selected folder to new source folder
    st.session_state['document_selected'] = my_document_name_selected
    logger.info("Executed document_selector()")

    return my_document_name_selected


def check_vectordb(my_querier: Querier,
                   my_folder_name_selected: str,
                   my_folder_path_selected: str,
                   my_vecdb_folder_path_selected: str) -> None:
    """
    checks if the vector database exists for the selected document folder, with the given settings
    If not, it creates the vector database

    Parameters
    ----------
    my_querier : Querier
        the querier object
    my_folder_name_selected : str
        the name of the selected folder
    my_folder_path_selected : str
        the path of the selected folder
    my_vecdb_folder_path_selected : str
        the name of the associated vector database
    """
    # If a folder is chosen that is not equal to the last known source folder
    if folder_name_selected != st.session_state['folder_selected']:
        # set session state of is_GO_clicked to False (will be set to True when OK button is clicked)
        st.session_state['is_GO_clicked'] = False
        # clear all chat messages on screen and in Querier object
        st.session_state['messages'] = []
        my_querier.clear_history()
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given
    # in settings.py, create it first
    if not os.path.exists(my_vecdb_folder_path_selected):
        logger.info("Creating vectordb")
        my_spinner_message = f'''Creating vector database for folder {my_folder_name_selected}.
                                 Depending on the size, this may take a while. Please wait...'''
    else:
        logger.info("Updating vectordb")
        my_spinner_message = f'''Checking if vector database needs an update for folder {my_folder_name_selected}.
                                 This may take a while, please wait...'''
    with st.spinner(my_spinner_message):
        ingester = Ingester(my_folder_name_selected,
                            my_folder_path_selected,
                            my_vecdb_folder_path_selected)
        ingester.ingest()

    # create a new chain based on the new source folder
    my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected)
    # set session state of selected folder to new source folder
    st.session_state['folder_selected'] = my_folder_name_selected
    logger.info("Executed check_vectordb")


def handle_query(my_folder_path_selected: str,
                 my_querier: Querier,
                 my_prompt: str,
                 my_document_selection: list,
                 my_folder_name_selected: str,
                 my_vecdb_folder_path_selected: str) -> None:
    """
    creates an answer to the user's prompt by invokeing the defined chain

    Parameters
    ----------
    my_folder_path_selected : str
        path of selected document folder
    my_querier : Querier
        querier object
    my_prompt : str
        user prompt
    document_selection : List[str]
        selected document(s)
    my_folder_name_selected : str
        selected document folder
    my_vecdb_folder_path_selected : str
        vector database associated with selected document folder
    """
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(my_prompt)
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": my_prompt})
    with st.spinner("Thinking..."):
        # Generate a response
        if 'All' not in my_document_selection:
            # create a filter for the selected documents
            my_filter = {'filename': document_name for document_name in my_document_selection}
            logger.info(f'Document Selection: {filter}')
            my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected, search_filter=my_filter)
        else:
            my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected)
        response = my_querier.ask_question(my_prompt)
    # Display the response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    # Add the response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response["answer"]})
    if len(response["source_documents"]) > 0:
        with st.expander("Paragraphs used for answer"):
            for i, document in enumerate(response["source_documents"]):
                filename = document.metadata['filename']
                docpath = os.path.join(my_folder_path_selected, filename)
                pagenr = document.metadata['page_number']
                content = document.page_content
                if filename.endswith(".pdf"):
                    exp_textcol, _, exp_imgcol = st.columns([0.3, 0.1, 0.6])
                else:
                    exp_textcol, _ = st.columns([0.9, 0.1])
                with exp_textcol:
                    # add 1 to metadata page_number because that starts at 0
                    st.write(f"**file: {filename}, page {pagenr + 1}**")
                    st.write(f"{document.page_content}")
                if filename.endswith(".pdf"):
                    with exp_imgcol:
                        doc = fitz.open(docpath)
                        page = doc.load_page(pagenr)
                        rects = page.search_for(content)
                        for rect in rects:
                            page.add_highlight_annot(rect)
                        # save image of page with highlighted text, zoom factor 2 in each dimension
                        zoom_x = 2
                        zoom_y = 2
                        mat = fitz.Matrix(zoom_x, zoom_y)
                        pix = page.get_pixmap(matrix=mat)
                        # store image as a PNG
                        imgfile = f"{docpath}-ch{i}.png"
                        pix.save(imgfile)
                        st.image(imgfile)
                st.divider()
    else:
        logger.info("No source documents found relating to the question")
    logger.info("Executed handle_query(querier, prompt)")


@st.cache_data
def initialize_page() -> None:
    """
    Initializes the main page with a page header and app info
    Also prepares the sidebar with folder list
    """
    # Custom CSS to have white expander background
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: white;
            color: black; # Expander header color
        }
        .streamlit-expanderContent {
            background-color: white;
            color: black; # Expander content color
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    logo_image = Image.open(settings.APP_LOGO)
    st.sidebar.image(logo_image, width=250)
    # # Sidebar text for folder selection
    # st.sidebar.title("Select folder")

    _, col2, _, col4 = st.columns([0.2, 0.4, 0.1, 0.3])
    with col2:
        st.header(settings.APP_HEADER)
    with col4:
        with st.expander("User manual"):
            # read app explanation from file explanation.txt
            with open(file=settings.APP_INFO, mode="r", encoding="utf8") as f:
                explanation = f.read()
            st.markdown(body=explanation, unsafe_allow_html=True)
            st.image("./images/multilingual.png")
    # set session state default for messages to fight hallucinations
    # st.session_state.setdefault('messages', [{"role": "system", "content": "You are a helpful assistant.
    logger.info("Executed initialize_page()")


def initialize_session_state() -> None:
    """
    Initialize the session state variables for control
    """
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""
    if 'document_selected' not in st.session_state:
        st.session_state['document_selected'] = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


@st.cache_resource
def initialize_querier():
    """
    Create a Querier object

    Returns
    -------
    Querier
        Querier object
    """
    my_querier = Querier()
    logger.info("Executed initialize_querier()")

    return my_querier


def set_page_config() -> None:
    """
    page configuration
    """
    st.set_page_config(page_title="Chat with your documents",
                       page_icon=':books:',
                       layout='wide',
                       initial_sidebar_state='auto')
    logger.info("\nExecuted set_page_config()")


# ### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()
# initialize page, executed only once per session
initialize_page()
# create list of content folders
source_folders_available = folderlist_creator()
# initialize session state variables
initialize_session_state()
# creation of Querier object, executed only once per session
querier = initialize_querier()
# chosen folder and associated vector database
folder_name_selected, folder_path_selected, vecdb_folder_path_selected = folder_selector(source_folders_available)
# available and selected documents
document_names = documentlist_creator(folder_path_selected)
document_selection = document_selector(document_names)

# create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True
st.sidebar.button("GO", type="primary", on_click=click_go_button)

# only start a conversation when a folder is selected and selection is confirmed with "GO" button
if st.session_state['is_GO_clicked']:
    # clear chat history any time GO is clicked
    st.session_state['messages'] = []
    querier.clear_history()
    # create or update vector database if necessary
    check_vectordb(querier, folder_name_selected, folder_path_selected, vecdb_folder_path_selected)
    summary_type = st.sidebar.radio(
        "Start with summary?",
        ["No", "Short", "Long", "Hybrid"],
        captions=["", "Quick but lower quality", "Slow but higher quality", "Hybrid of quick and slow"],
        index=0)
    # if one of the options is chosen
    if summary_type in ["Short", "Long", "Hybrid"]:
        # show the summary at the top of the screen
        create_and_show_summary(summary_type, folder_path_selected,
                                folder_name_selected,
                                vecdb_folder_path_selected,
                                document_selection)

    # show button "Clear Conversation"
    clear_messages_button = st.button(label="Clear Conversation", key="clear")

    # if button "Clear Conversation" is clicked
    if clear_messages_button:
        # clear all chat messages on screen and in Querier object
        # NB: session state of "is_GO_clicked" and "folder_selected" remain unchanged
        st.session_state['messages'] = []
        querier.clear_history()
        logger.info("Clear Conversation button clicked")

    # display chat messages from history
    display_chat_history()

    # react to user input if a question has been asked
    if prompt := st.chat_input("Your question"):
        handle_query(folder_path_selected, querier, prompt, document_selection, folder_name_selected, vecdb_folder_path_selected)
