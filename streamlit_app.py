"""
Streamlit User Interface for chatting with documents
"""
from typing import List
import os
import fitz
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


def click_exit_button():
    """
    Sets session state of EXIT button clicked to True
    """
    st.session_state['is_EXIT_clicked'] = True


@st.cache_data
def create_and_show_summary(my_summary_type: str,
                            my_folder_path_selected: str,
                            my_selected_documents: List[str]) -> None:
    """
    Creates or loads a summary of the chosen document(s) and shows it in the UI

    Parameters
    ----------
    my_summary_type : str
        chosen summary type, either "Short" or "Long"
    my_folder_path_selected : str
        path of content folder
    my_selected_documents : List[str]
        list of selected documents
    """
    summarization_method = "map_reduce"
    if my_summary_type == "Long":
        summarization_method = "refine"

    logger.info(f"Starting create_and_show_summary() with summarization method {summarization_method}")
    # create subfolder for storage of summaries if not existing
    ut.create_summaries_folder(my_folder_path_selected)
    # get relevant models
    confidential = False  # temporary
    my_llm_provider, my_llm_model, _, _ = ut.get_relevant_models(summary=True,
                                                                 private=confidential)
    summarizer = Summarizer(content_folder_path=my_folder_path_selected,
                            summarization_method=summarization_method,
                            text_splitter_method=settings.SUMMARY_TEXT_SPLITTER_METHOD,
                            chunk_size=settings.SUMMARY_CHUNK_SIZE,
                            chunk_overlap=settings.SUMMARY_CHUNK_OVERLAP,
                            llm_provider=my_llm_provider,
                            llm_model=my_llm_model)

    # for each selected file in content folder
    with st.expander(label=f"{my_summary_type} summary", expanded=True):
        first_summary = True
        files_in_folder = ut.get_relevant_files_in_folder(my_folder_path_selected)
        if my_selected_documents == ["All"]:
            my_selected_documents = files_in_folder
        for file in files_in_folder:
            if file in my_selected_documents:
                file_name, _ = os.path.splitext(file)
                summary_name = os.path.join(my_folder_path_selected, "summaries",
                                            str(file_name) + "_" + str.lower(summarization_method) + ".txt")
                # if summary does not exist yet, create it
                if not os.path.isfile(summary_name):
                    my_spinner_message = f'''Creating {my_summary_type.lower()} summary for {file}.\n
                    Depending on the size of the file and the type of summary, this may take a while. Please wait...'''
                    with st.spinner(my_spinner_message):
                        summarizer.summarize_file(file)
                # show summary
                if not first_summary:
                    st.divider()
                with open(file=summary_name, mode="r", encoding="utf8") as f:
                    st.write(f"**{file}:**\n")
                    st.write(f.read())
                first_summary = False
    logger.info(f"Finished create_and_show_summary() with summarization method {summarization_method}")


def display_chat_history(my_folder_path_selected) -> None:
    """
    Shows the complete chat history with source documents displayed after each assistant response.

    Parameters
    ----------
    folder_path_selected : str
        path of selected document folder
    """
    for i, message in enumerate(st.session_state['messages']):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
            # Show source documents right after the assistant's response
            if "sources" in message and message["sources"]:
                display_sources(sources=message["sources"],
                                my_folder_path_selected=my_folder_path_selected,
                                question_number=i)


@st.cache_data
def documentlist_creator(my_folder_path_selected: str) -> List[str]:
    """
    Creates a list of document names (without path), available in the selected folder path

    Parameters
    ----------
    my_folder_path_selected : str
        the selected document folder path

    Returns
    -------
    List[str]
        list of available and valid document names
    """
    relevant_files_in_folder = ut.get_relevant_files_in_folder(my_folder_path_selected)
    # Show option "All" only if there are multiple relevant documents in a folder
    if len(relevant_files_in_folder) > 1:
        relevant_files_in_folder.insert(0, "All")
    logger.info("Executed documentlist_creator()")

    return relevant_files_in_folder


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
    my_document_names_selected = st.sidebar.multiselect(label="***SELECT ANY / ALL FILES***",
                                                        options=documents,
                                                        default=documents[0],
                                                        key='document_selector')
    logger.info(f"document_names_selected is now {my_document_names_selected}")
    logger.info("Executed document_selector()")

    return my_document_names_selected


def check_vectordb(my_querier: Querier,
                   my_folder_name_selected: str,
                   my_folder_path_selected: str,
                   my_documents_selected: List[str],
                   my_vecdb_folder_path_selected: str,
                   my_embeddings_provider: str,
                   my_embeddings_model: str) -> None:
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
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given
    # in settings.py, create it first
    if not os.path.exists(my_vecdb_folder_path_selected):
        ut.create_vectordb_folder(my_folder_path_selected)
        logger.info("Creating vectordb")
        my_spinner_message = f'''Creating vector database for folder {my_folder_name_selected}.
                                 Depending on the size, this may take a while. Please wait...'''
    else:
        logger.info("Updating vectordb")
        my_spinner_message = f'''Checking if vector database needs an update for folder {my_folder_name_selected}.
                                 This may take a while, please wait...'''
    with st.spinner(my_spinner_message):
        ingester = Ingester(collection_name=my_folder_name_selected,
                            content_folder=my_folder_path_selected,
                            document_selection=my_documents_selected,
                            vecdb_folder=my_vecdb_folder_path_selected,
                            embeddings_provider=my_embeddings_provider,
                            embeddings_model=my_embeddings_model)
        ingester.ingest()

    # create a new chain based on the new source folder
    my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected)
    # set session state of selected folder to new source folder
    st.session_state['folder_selected'] = my_folder_name_selected
    logger.info("Executed check_vectordb")


def display_sources(sources: List[str], my_folder_path_selected: str, question_number: int) -> None:
    """
    Displays the source documents used for the answer

    Parameters
    ----------
    sources : List[str]
        list of source documents
    my_folder_path_selected : str
        path of selected document folder
    question_number : int
        number of the question asked by the user to reduce the number of images created
        (in later versions it can be used to check if the images are already created and stored)
    """
    if len(sources) > 0:
        with st.expander("Paragraphs used for answer"):
            # group sources by filename and page number
            # in order to show the same page with multiple highlights only once
            grouped_sources = {}
            for doc in sources:
                key = (doc.metadata['filename'], doc.metadata['page_number'])
                if key not in grouped_sources:
                    grouped_sources[key] = []
                grouped_sources[key].append(doc)

            for i, (_, documents) in enumerate(grouped_sources.items()):
                filename = documents[0].metadata['filename']
                if filename.endswith(".docx"):
                    docpath = os.path.join(my_folder_path_selected, "conversions", filename + ".pdf")
                else:
                    docpath = os.path.join(my_folder_path_selected, filename)
                pagenr = documents[0].metadata['page_number']
                if (filename.endswith(".pdf")) or (filename.endswith(".docx")):
                    exp_textcol, _, exp_imgcol = st.columns([0.3, 0.1, 0.6])
                else:
                    exp_textcol, _ = st.columns([0.9, 0.1])
                with exp_textcol:
                    for document in documents:
                        # add 1 to metadata page_number because that starts at 0
                        st.write(f"**file: {filename}, page {pagenr + 1}**")
                        st.write(f"{document.page_content}")
                if (filename.endswith(".pdf")) or (filename.endswith(".docx")):
                    with exp_imgcol:
                        doc = fitz.open(docpath)
                        page = doc.load_page(pagenr)
                        # highlight all occurrences of the search string in the page
                        # there might be multiple occurrences of the same page with different chunks
                        for document in documents:
                            for rect in page.search_for(document.page_content):
                                page.add_highlight_annot(rect)
                        # save image of page with highlighted text, zoom factor 2 in each dimension
                        zoom_x = 2
                        zoom_y = 2
                        mat = fitz.Matrix(zoom_x, zoom_y)
                        pix = page.get_pixmap(matrix=mat)
                        # store image as a PNG
                        # question_number//2 + 1 is used to reduce the number of images created
                        imgfile = f"{docpath}-q{question_number//2 + 1}-ch{i}.png"
                        pix.save(imgfile)
                        st.image(imgfile)
                st.divider()


def handle_query(my_folder_path_selected: str,
                 my_querier: Querier,
                 my_prompt: str,
                 my_document_selection: List[str],
                 my_folder_name_selected: str,
                 my_vecdb_folder_path_selected: str,
                 question_number: int) -> None:
    """
    creates an answer to the user's prompt by invoking the defined chain

    Parameters
    ----------
    my_folder_path_selected : str
        path of selected document folder
    my_querier : Querier
        querier object
    my_prompt : str
        user prompt
    my_document_selection : List[str]
        selected document(s)
    my_folder_name_selected : str
        selected document folder
    my_vecdb_folder_path_selected : str
        vector database associated with selected document folder
    question_number : int
        number of the question asked by the user
    """
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(my_prompt)

    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": my_prompt})

    with st.spinner("Processing..."):
        # Generate a response
        if 'All' not in my_document_selection:
            # create a filter for the selected documents
            my_filter = {'filename': {'$in': my_document_selection}}
            logger.info(f'Document Selection filter: {my_filter}')
            my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected, search_filter=my_filter)
        else:
            my_querier.make_chain(my_folder_name_selected, my_vecdb_folder_path_selected)
        response = my_querier.ask_question(my_prompt)
    # Display the response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    # Add the response to chat history and also the source documents used for the answer
    st.session_state['messages'].append({"role": "assistant",
                                         "content": response["answer"],
                                         "sources": response["source_documents"]})

    # show sources for the answer
    if response["source_documents"]:
        display_sources(sources=response["source_documents"],
                        my_folder_path_selected=my_folder_path_selected,
                        question_number=question_number)
    else:
        logger.info("No source documents found relating to the question")
    logger.info("Executed handle_query(querier, prompt)")


@st.cache_data
def initialize_page() -> None:
    """
    Initializes the main page with a page header and app info
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
    _, col2, _ = st.columns([0.4, 0.2, 0.4])
    with col2:
        st.header(settings.APP_HEADER)
    with st.expander("User manual"):
        # read app explanation from file explanation.txt
        with open(file=settings.APP_INFO, mode="r", encoding="utf8") as f:
            explanation = f.read()
        st.markdown(body=explanation, unsafe_allow_html=True)

    logger.info("Executed initialize_page()")


@st.cache_data
def initialize_logo() -> None:
    """
    Initializes the main page with a page header and app info
    """
    logo_image = Image.open(settings.APP_LOGO)
    st.sidebar.image(logo_image, width=250)
    logger.info("Executed initialize_logo()")


@st.cache_data
def initialize_session_state() -> None:
    """
    Initializes the session state variables for control
    """
    if 'is_GO_clicked' not in st.session_state:
        st.session_state['is_GO_clicked'] = False
    if 'is_EXIT_clicked' not in st.session_state:
        st.session_state['is_EXIT_clicked'] = False
    if 'is_summary_clicked' not in st.session_state:
        st.session_state['is_summary_clicked'] = ""
    if 'folder_selected' not in st.session_state:
        st.session_state['folder_selected'] = ""
    if 'documents_selected' not in st.session_state:
        st.session_state['documents_selected'] = ""
    if 'confidential' not in st.session_state:
        st.session_state['confidential'] = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


# @st.cache_resource
def initialize_querier(my_llm_provider: str,
                       my_llm_model: str,
                       my_embeddings_provider: str,
                       my_embeddings_model: str) -> Querier:
    """
    Create a Querier object

    Parameters
    ----------
    my_llm_provider : str
        chosen llm provider
    my_llm_model : str
        chosen llm model
    my_embeddings_provider : str
        chosen embeddings provider
    my_embeddings_model : str
        chosen embeddings model

    Returns
    -------
    Querier
        Querier object
    """
    my_querier = Querier(llm_provider=my_llm_provider,
                         llm_model=my_llm_model,
                         embeddings_provider=my_embeddings_provider,
                         embeddings_model=my_embeddings_model)
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
# Create button to exit the application. This button sets session_state['is_EXIT_clicked'] to True
st.sidebar.button("EXIT", type="primary", on_click=click_exit_button)
# initialize logo, executed only once per session
initialize_logo()
# initialize session state variables
initialize_session_state()
# allow user to set the path to the document folder
folder_path_selected = st.sidebar.text_input(label="***ENTER THE DOCUMENT FOLDER PATH***",
                                             help="""Please enter the full path e.g. Y:/User/troosts/chatpbl/...""")
if st.session_state['is_EXIT_clicked']:
    ut.exit_ui()
if folder_path_selected != "":
    # get folder name with docs
    folder_name_selected = os.path.basename(folder_path_selected)
    # available and selected documents
    document_names = documentlist_creator(folder_path_selected)
    document_selection = document_selector(document_names)
    # create checkbox to indicate whether chosen documents are private or not. Default is not checked
    # confidential = st.sidebar.checkbox(label="confidential", help="check in case of private documents")
    confidential = False
    # get relevant models
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                            private=confidential)
    # creation of Querier object, executed only once per session
    querier = initialize_querier(my_llm_provider=llm_provider,
                                 my_llm_model=llm_model,
                                 my_embeddings_provider=embeddings_provider,
                                 my_embeddings_model=embeddings_model)

    # If a different folder or (set of) document(s) is chosen,
    # clear querier history and
    # set the go button session state 'is_go_clicked' to False
    if ((folder_name_selected != st.session_state['folder_selected']) or
       (document_selection != st.session_state['documents_selected'])):
        querier.clear_history()
        st.session_state['is_GO_clicked'] = False
    st.session_state['folder_selected'] = folder_name_selected
    st.session_state['documents_selected'] = document_selection

    # clear querier history if a switch in confidentiality is made
    if confidential != st.session_state['confidential']:
        querier.clear_history()
        st.session_state['is_GO_clicked'] = False
    st.session_state['confidential'] = confidential

    # create radio button group for summarization
    summary_type = st.sidebar.radio(label="Start with summary?",
                                    options=["No", "Short", "Long"],
                                    captions=["", "Quicker and shorter", "Slower but more extensive"],
                                    index=0)
    # when a different summary choice is made, set the go button session state 'is_go_clicked' to False
    if summary_type != st.session_state['is_summary_clicked']:
        st.session_state['is_GO_clicked'] = False
    st.session_state['is_summary_clicked'] = summary_type

    # create button to confirm folder selection. This button sets session_state['is_GO_clicked'] to True when clicked
    st.sidebar.button(label="GO", type="primary", on_click=click_go_button,
                      help="show the prompt bar at the bottom of the screen to take questions")

    # only start a conversation when a folder is selected and selection is confirmed with "GO" button
    if st.session_state['is_GO_clicked']:
        logger.info("GO button is clicked")
        if len(document_selection) > 0:
            # get relevant models
            _, _, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                 private=confidential)

            # determine name of associated vector database
            vecdb_folder_path = ut.create_vectordb_path(content_folder_path=folder_path_selected,
                                                        embeddings_provider=embeddings_provider,
                                                        embeddings_model=embeddings_model)
            # create or update vector database if necessary
            check_vectordb(my_querier=querier,
                           my_folder_name_selected=folder_name_selected,
                           my_folder_path_selected=folder_path_selected,
                           my_documents_selected=document_selection,
                           my_vecdb_folder_path_selected=vecdb_folder_path,
                           my_embeddings_provider=embeddings_provider,
                           my_embeddings_model=embeddings_model)
            # if one of the options is chosen
            if summary_type in ["Short", "Long"]:
                # show the summary at the top of the screen
                create_and_show_summary(my_summary_type=summary_type,
                                        my_folder_path_selected=folder_path_selected,
                                        my_selected_documents=document_selection)
            # show button "Clear Conversation"
            clear_messages_button = st.button(label="Clear Conversation", key="clear")
            # if button "Clear Conversation" is clicked
            if clear_messages_button:
                # clear all chat messages on screen and in Querier object
                # NB: session state of "is_GO_clicked" and "folder_selected" remain unchanged
                st.session_state['messages'] = []
                querier.clear_history()
                st.session_state['is_GO_clicked'] = False
                logger.info("Clear Conversation button clicked")
            # display chat messages from history
            # path is needed to show source documents after the assistant's response
            display_chat_history(folder_path_selected)
            # react to user input if a question has been asked
            prompt = st.chat_input("Your question")
            if prompt:
                handle_query(my_folder_path_selected=folder_path_selected,
                             my_querier=querier,
                             my_prompt=prompt,
                             my_document_selection=document_selection,
                             my_folder_name_selected=folder_name_selected,
                             my_vecdb_folder_path_selected=vecdb_folder_path,
                             question_number=len(st.session_state['messages']))
        else:
            st.write("Please choose one or more documents")
