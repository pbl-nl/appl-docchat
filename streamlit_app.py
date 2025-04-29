"""
Streamlit User Interface for chatting with documents
"""
from typing import List, Dict
import os
import fitz
import streamlit as st
from PIL import Image
from loguru import logger
import re
import traceback
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
from summarize.summarizer import Summarizer
import settings
import settings_template
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
    if not st.session_state['is_in_memory']:
        ut.create_summaries_folder(my_folder_path_selected)
    elif summarization_method not in st.session_state['summary_storage']:
        st.session_state['summary_storage'][summarization_method] = {}
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
                            llm_model=my_llm_model,
                            in_memory=st.session_state['is_in_memory'])

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
                    if st.session_state['is_in_memory'] and (file not in
                                                             st.session_state['summary_storage'][summarization_method]):
                        with st.spinner(my_spinner_message):
                            st.session_state['summary_storage'][summarization_method][file] = \
                                                                                        summarizer.summarize_file(file)
                    else:
                        with st.spinner(my_spinner_message):
                            summarizer.summarize_file(file)
                # show summary
                if not first_summary:
                    st.divider()
                if st.session_state['is_in_memory'] and (file in
                                                         st.session_state['summary_storage'][summarization_method]):
                    st.write(f"**{file}:**\n")
                    st.write(st.session_state['summary_storage'][summarization_method][file])
                else:
                    # show summary from file
                    st.write(f"**{file}:**\n")
                    with open(file=summary_name, mode="r", encoding="utf8") as f:
                        st.write(f.read())
                first_summary = False
    logger.info(f"Finished create_and_show_summary() with summarization method {summarization_method}")


def display_chat_history(my_folder_path_selected: str) -> None:
    """
    Shows the complete chat history with source documents displayed after each assistant response.

    Parameters
    ----------
    my_folder_path_selected : str
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
        path of selected document folder

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
                   my_embeddings_model: str,
                   my_text_splitter_method: str,
                   my_retriever_type: str,
                   my_chunk_size: int,
                   my_chunk_overlap: int,
                   my_text_splitter_method_child: str,
                   my_chunk_size_child: int,
                   my_chunk_overlap_child: int) -> None:
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
    my_embeddings_provider : str
        the chosen embeddings provider
    my_embeddings_model : str
        the chosen embeddings model
    my_text_splitter_method : str
        the chosen text splitter method
    my_retriever_type : str
        the chosen retriever type
    my_chunk_size : int
        the chosen chunk size
    my_chunk_overlap : int
        the chosen chunk overlap
    my_text_splitter_method_child : str
        the chosen text splitter method to create child chunks
    my_chunk_size_child : int
        the chosen chunk size for child chunks
    my_chunk_overlap_child : int
        the chosen chunk overlap for child chunks
    """
    # check whether the selected folder is the same as the last selected folder
    if my_vecdb_folder_path_selected != st.session_state['vecdb_folder_path_selected']:
        st.session_state['vector_store'] = None  # reset vector store
        st.session_state['vecdb_folder_path_selected'] = my_vecdb_folder_path_selected
    # When the associated vector database of the chosen content folder doesn't exist with the settings as given
    # in settings.py, create it first
    if st.session_state['is_in_memory']:
        my_vecdb_folder_path_selected = None
        my_spinner_message = f'''Error creating/accessing vector database for folder {my_folder_name_selected}.
                                    In-memory vector database will be created.
                                    Depending on the size, this may take a while. Please wait...'''
    else:
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
        # create ingester object
        ingester = Ingester(collection_name=my_folder_name_selected,
                            content_folder=my_folder_path_selected,
                            document_selection=my_documents_selected,
                            vecdb_folder=my_vecdb_folder_path_selected,
                            embeddings_provider=my_embeddings_provider,
                            embeddings_model=my_embeddings_model,
                            text_splitter_method=my_text_splitter_method,
                            retriever_type=my_retriever_type,
                            chunk_size=my_chunk_size,
                            chunk_overlap=my_chunk_overlap,
                            text_splitter_method_child=my_text_splitter_method_child,
                            chunk_size_child=my_chunk_size_child,
                            chunk_overlap_child=my_chunk_overlap_child,
                            vector_store=st.session_state['vector_store'],
                            in_memory=st.session_state['is_in_memory'])
        if ut.check_size(my_folder_path_selected, my_documents_selected) <= settings_template.MAX_INGESTION_SIZE:
            ingester.ingest()
            st.session_state['file_size_error'] = False
            st.session_state['vector_store'] = ingester.vector_store
        else:
            st.error(f"Size of the files to be ingested exceeds the limit of {settings_template.MAX_INGESTION_SIZE} MB")
            st.session_state['file_size_error'] = True

    # create a new chain based on the new source folder
    my_querier.vector_store = st.session_state['vector_store']
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
        ordinal number of the question asked by the user
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
                if (filename.endswith(".pdf")) or\
                   ((filename.endswith(".docx") and not st.session_state['is_in_memory'])):
                    # question_number//2 + 1 is used due to human and chatbot interaction
                    imgname = f"{docpath}-q{question_number//2 + 1}-ch{i}.png"
                    if imgname not in st.session_state['source_image']:
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
                            # store image as a binary string
                            img_bytes = pix.tobytes()
                            from io import BytesIO
                            img_io = BytesIO(img_bytes)
                            st.session_state['source_image'][imgname] = img_io
                            st.image(img_io)
                    else:
                        with exp_imgcol:
                            st.image(st.session_state['source_image'][imgname])
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
        # reload chat history if it is cleared by a change in settings
        if (len(st.session_state['chat_history']) > 0) and (my_querier.chat_history == []):
            my_querier.chat_history = st.session_state['chat_history']
            logger.info("chat history retained")
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

    # to retain chat history in case of a change in settings
    st.session_state['chat_history'] = my_querier.chat_history.copy()


@st.cache_data
def initialize_page() -> None:
    """
    Initializes the main page with a page header and app info
    """

    # Custom CSS to have white expander background and fixed chat input
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: white;
            color: black;
        }
        .streamlit-expanderContent {
            background-color: white;
            color: black;
        }
        /* Fixed chat input */
        .st-key-chat_input {
            position: fixed !important;
            bottom: 0 !important;
            padding: 1rem 0 2rem 0 !important;
            z-index: 999999 !important;
            margin: 0 !important;
        }
        /* Fixed save button */
        .st-key-save_settings {
            position: fixed !important;
            bottom: 10% !important;
            right: 20% !important;
            padding: 0.5rem !important;  /* reduced padding */
            z-index: 999999 !important;
            margin: 0 !important;
            width: 0 !important;      /* specific width */
            height: 0 !important;     /* specific height */
            /* center content */
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
    # chat history is stored in session state to retain it in case of a change in settings
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'file_size_error' not in st.session_state:
        st.session_state['file_size_error'] = False
    if 'source_image' not in st.session_state:
        st.session_state['source_image'] = {}
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'vecdb_folder_path_selected' not in st.session_state:
        st.session_state['vecdb_folder_path_selected'] = ""
    if 'summary_storage' not in st.session_state:
        st.session_state['summary_storage'] = {}
    if 'is_in_memory' not in st.session_state:
        st.session_state['is_in_memory'] = False
    initialize_settings_state()


# @st.cache_resource
def initialize_querier(my_llm_provider: str,
                       my_llm_model: str,
                       my_embeddings_provider: str,
                       my_embeddings_model: str,
                       my_retriever_type: str,
                       my_rerank: bool,
                       my_chunk_k: int,
                       my_search_type: str,
                       my_score_threshold: float) -> Querier:
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
    my_retriever_type : str
        chosen retriever type
    my_rerank : bool
        chosen rerank option
    my_chunk_k : int
        chosen max nr of chunks to retrieve
    my_search_type : str
        chosen search type
    my_score_threshold : float
        chosen score threshold

    Returns
    -------
    Querier
        Querier object
    """
    my_querier = Querier(llm_provider=my_llm_provider,
                         llm_model=my_llm_model,
                         embeddings_provider=my_embeddings_provider,
                         embeddings_model=my_embeddings_model,
                         retriever_type=my_retriever_type,
                         rerank=my_rerank,
                         chunk_k=my_chunk_k,
                         search_type=my_search_type,
                         score_threshold=my_score_threshold)
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


def get_provider_models() -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dictionary of available models for each provider type
    """
    return {
        'embeddings': {
            'openai': [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ],
            'huggingface': [
                "all-mpnet-base-v2"
            ],
            'ollama': [
                "llama3",
                "nomic-embed-text"
            ],
            'azureopenai': [
                "text-embedding-ada-002",
                "text-embedding-3-large"
            ]
        },
        'llm': {
            'openai': [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4o"
            ],
            'huggingface': [
                "meta-llama/Llama-2-7b-chat-hf",
                "google/flan-t5-base"
            ],
            'ollama': [
                "llama3",
                "orca-mini",
                "zephyr"
            ],
            'azureopenai': [
                "gpt-35-turbo",
                # "gpt-4",
                "gpt-4o"
            ]
        }
    }


def initialize_settings_state():
    """Initialize session state for settings if not already present"""
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'TEXT_SPLITTER_METHOD': settings_template.TEXT_SPLITTER_METHOD,
            'EMBEDDINGS_PROVIDER': settings_template.EMBEDDINGS_PROVIDER,
            'EMBEDDINGS_MODEL': settings_template.EMBEDDINGS_MODEL,
            'RETRIEVER_TYPE': settings_template.RETRIEVER_TYPE,
            'RERANK': settings_template.RERANK,
            'LLM_PROVIDER': settings_template.LLM_PROVIDER,
            'LLM_MODEL': settings_template.LLM_MODEL,
            'CHUNK_K': settings_template.CHUNK_K,
            'CHUNK_SIZE': settings_template.CHUNK_SIZE,
            'CHUNK_OVERLAP': settings_template.CHUNK_OVERLAP,
            'TEXT_SPLITTER_METHOD_CHILD': settings_template.TEXT_SPLITTER_METHOD_CHILD,
            'CHUNK_SIZE_CHILD': settings_template.CHUNK_SIZE_CHILD,
            'CHUNK_OVERLAP_CHILD': settings_template.CHUNK_OVERLAP_CHILD,
            'SEARCH_TYPE': settings_template.SEARCH_TYPE,
            'SCORE_THRESHOLD': settings_template.SCORE_THRESHOLD,
        }


def read_description():
    """Read the description from the file and return it"""
    with open(file="settings_template.py", mode="r", encoding="utf8") as f:
        description = f.read()
    return description


def find_setting_description(content: str, setting_name: str) -> str:
    """
    Find all consecutive commented lines that appear immediately before a setting variable
    and combine them into a single line.

    Args:
        content (str): The content of settings_template.py
        setting_name (str): The name of the setting to find

    Returns:
        str: Combined consecutive commented lines before the setting, without the setting line
    """
    pattern = rf"(?:\n|^)((?:#[^\n]*\n)+)(?={setting_name.strip()}\s*=)"
    match = re.search(pattern, content)
    if match:
        comments = match.group(1).rstrip()
        # Filter out any separator comments (those with only # and special characters)
        relevant_comments = [line for line in comments.split('\n')
                             if not re.match(r'^#[\s*#]*$', line.strip())]
        return ' '.join(line.strip() for line in relevant_comments)
    return ""


def get_settings_descriptions():
    """
    Read settings_template.py and extract descriptions for all settings.
    Returns a dictionary with setting names as keys and their descriptions as values.
    """
    content = read_description()

    # List of settings to find descriptions for
    settings = [
        "TEXT_SPLITTER_METHOD",
        "CHUNK_K",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "EMBEDDINGS_PROVIDER",
        "EMBEDDINGS_MODEL",
        "RETRIEVER_TYPE",
        "TEXT_SPLITTER_METHOD_CHILD",
        "CHUNK_SIZE_CHILD",
        "CHUNK_OVERLAP_CHILD",
        "RERANK",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "SEARCH_TYPE",
        "SCORE_THRESHOLD"
    ]

    # Dictionary to store setting descriptions
    descriptions = {}

    # Get description for each setting
    for setting in settings:
        description = find_setting_description(content, setting)
        descriptions[setting] = "<br>".join(desc.replace("# ", "").lstrip("# ") for desc in description.split("# "))[4:]

    return descriptions


def selectbox(label, options, key, description, subheader=None):
    if subheader:
        st.subheader(subheader)

    if key not in st.session_state:
        st.session_state[key] = st.session_state.settings[key]

    index = options.index(st.session_state[key]) if st.session_state[key] in options else 0

    # with col1:
    selected = st.selectbox(
        label=label,
        options=options,
        index=index,
        key=key
    )
    with st.expander("Description"):
        st.markdown(f"<p style='font-size:18px;'>{description}</p>", unsafe_allow_html=True)
    # st.divider()
    return selected


def number_input(label, min_value, max_value, key, description, subheader=None):
    if subheader:
        st.subheader(subheader)

    if key not in st.session_state:
        st.session_state[key] = st.session_state.settings[key]

    default_value = max(min_value, min(max_value, st.session_state[key]))

    # with col1:
    number = st.number_input(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        key=key
    )
    with st.expander("Description"):
        st.markdown(f"<p style='font-size:18px;'>{description}</p>", unsafe_allow_html=True)

    return number


def checkbox(label, key, description, subheader=None):
    if subheader:
        st.subheader(subheader)

    if key not in st.session_state:
        st.session_state[key] = st.session_state.settings[key]

    # with col1:
    checked = st.checkbox(
        label=label,
        value=st.session_state[key],
        key=key
    )
    with st.expander("Description"):
        st.markdown(f"<p style='font-size:18px;'>{description}</p>", unsafe_allow_html=True)

    return checked


def save_settings(selected_splitter, selected_embeddings_provider, selected_embedding_model, selected_retriever,
                  rerank_enabled, selected_llm_provider, selected_llm_model, chunk_k, chunk_size, chunk_overlap,
                  selected_splitter_child, chunk_size_child, chunk_overlap_child, search_type, score_threshold,
                  developer_mode):

    if not st.session_state['confidential']:
        st.session_state.settings.update({
            'TEXT_SPLITTER_METHOD': selected_splitter,
            'EMBEDDINGS_PROVIDER': selected_embeddings_provider,
            'EMBEDDINGS_MODEL': selected_embedding_model,
            'RETRIEVER_TYPE': selected_retriever,
            'RERANK': rerank_enabled,
            'LLM_PROVIDER': selected_llm_provider,
            'LLM_MODEL': selected_llm_model,
            'CHUNK_K': chunk_k,
            'CHUNK_SIZE': chunk_size,
            'CHUNK_OVERLAP': chunk_overlap,
            'TEXT_SPLITTER_METHOD_CHILD': selected_splitter_child,
            'CHUNK_SIZE_CHILD': chunk_size_child,
            'CHUNK_OVERLAP_CHILD': chunk_overlap_child,
            'SEARCH_TYPE': search_type,
            'SCORE_THRESHOLD': score_threshold
        })
        st.success("Settings saved successfully!")

        # Log current settings
        if developer_mode:
            st.write("Current Settings:")
            st.json(st.session_state.settings)
    else:
        st.error("Confidential mode is enabled. Settings cannot be changed.")


def text_processing_settings_tab(descriptions):
    # Text Splitter Method
    selected_splitter = selectbox(subheader="Text Processing",
                                  label="Text Splitter Method",
                                  options=["RecursiveCharacterTextSplitter", "NLTKTextSplitter"],
                                  key='TEXT_SPLITTER_METHOD',
                                  description=descriptions['TEXT_SPLITTER_METHOD'])

    # Chunk Size
    chunk_size = number_input(label="Chunk Size",
                              min_value=1,
                              max_value=2000,
                              key='CHUNK_SIZE',
                              description=descriptions['CHUNK_SIZE'])

    # Chunk Overlap
    chunk_overlap = number_input(label="Chunk Overlap",
                                 min_value=0,
                                 max_value=int(0.1 * chunk_size),
                                 key='CHUNK_OVERLAP',
                                 description=descriptions['CHUNK_OVERLAP'],)

    return selected_splitter, chunk_size, chunk_overlap


def embeddings_settings_tab(descriptions, provider_models, choose_llm_emb_provider):
    # TO BE REMOVED, ONCE OTHER EMBEDDINGS PROVIDERS ARE IMPLEMENTED
    if choose_llm_emb_provider:
        selected_embeddings_provider = selectbox(subheader="Embedding",
                                                 label="Embeddings Provider",
                                                 options=["openai", "huggingface", "ollama", "azureopenai"],
                                                 key='EMBEDDINGS_PROVIDER',
                                                 description=descriptions['EMBEDDINGS_PROVIDER'])
    else:
        selected_embeddings_provider = "azureopenai"

    # Get available models for selected provider
    available_embedding_models = provider_models['embeddings'][selected_embeddings_provider]

    # Default to first available model if current isn't available for selected provider
    current_emb_model = st.session_state.settings['EMBEDDINGS_MODEL']
    if current_emb_model not in available_embedding_models:
        current_emb_model = available_embedding_models[0]

    selected_embedding_model = selectbox(subheader=None if choose_llm_emb_provider else "Embedding",
                                         label="Embeddings Model",
                                         options=available_embedding_models,
                                         key='EMBEDDINGS_MODEL',
                                         description=descriptions['EMBEDDINGS_MODEL'])
    return selected_embeddings_provider, selected_embedding_model


def retrieve_settings_tab(descriptions, chunk_size):
    # Number of chunks
    chunk_k = number_input(subheader="Retrieval",
                           label="Chunk K",
                           min_value=1,
                           max_value=8,
                           key='CHUNK_K',
                           description=descriptions['CHUNK_K'])

    # Text Splitter Method
    search_type = selectbox(label="Search Type",
                            options=["similarity_score_threshold", "similarity"],
                            key='SEARCH_TYPE',
                            description=descriptions['SEARCH_TYPE'])

    score_threshold = st.session_state.settings.get('SCORE_THRESHOLD', None)
    if search_type == "similarity_score_threshold":
        score_threshold = number_input(label="Similarity Score Threshold",
                                       min_value=0.0,
                                       max_value=1.0,
                                       key='SCORE_THRESHOLD',
                                       description=descriptions['SCORE_THRESHOLD'])

    selected_retriever = selectbox(label="Retriever Type",
                                   options=["vectorstore", "hybrid", "parent"],
                                   key='RETRIEVER_TYPE',
                                   description=descriptions['RETRIEVER_TYPE'])

    selected_splitter_child, chunk_size_child, \
        chunk_overlap_child = [st.session_state.settings['TEXT_SPLITTER_METHOD_CHILD'],
                               st.session_state.settings['CHUNK_SIZE_CHILD'],
                               st.session_state.settings['CHUNK_OVERLAP_CHILD']]

    if selected_retriever == "parent":
        # Child Text Splitter Method
        selected_splitter_child = selectbox(label="Child Text Splitter Method",
                                            options=["RecursiveCharacterTextSplitter", "NLTKTextSplitter"],
                                            key='TEXT_SPLITTER_METHOD_CHILD',
                                            description=descriptions['TEXT_SPLITTER_METHOD_CHILD'])

        # Child Chunk Size
        chunk_size_child = number_input(label="Child Chunk Size",
                                        min_value=1,
                                        max_value=chunk_size,
                                        key='CHUNK_SIZE_CHILD',
                                        description=descriptions['CHUNK_SIZE_CHILD'])

        # Child Chunk Overlap
        chunk_overlap_child = number_input(label="Child Chunk Overlap",
                                           min_value=0,
                                           max_value=chunk_size_child-1,
                                           key='CHUNK_OVERLAP_CHILD',
                                           description=descriptions['CHUNK_OVERLAP_CHILD'],)

    # Reranking Configuration
    rerank_enabled = checkbox(label="Enable Reranking",
                              key='RERANK',
                              description=descriptions['RERANK'])

    return chunk_k, search_type, score_threshold, selected_retriever, selected_splitter_child, \
        chunk_size_child, chunk_overlap_child, rerank_enabled


def llm_settings_tab(descriptions, provider_models, choose_llm_emb_provider):
    if choose_llm_emb_provider:
        selected_llm_provider = selectbox(subheader="LLM",
                                          label="LLM Provider",
                                          options=["openai", "huggingface", "ollama", "azureopenai"],
                                          key='LLM_PROVIDER',
                                          description=descriptions['LLM_PROVIDER'])
    else:
        selected_llm_provider = "azureopenai"

    # Get available models for selected provider
    available_llm_models = provider_models['llm'][selected_llm_provider]

    # Default to first available model if current isn't available for selected provider
    current_llm_model = st.session_state.settings['LLM_MODEL']
    if current_llm_model not in available_llm_models:
        current_llm_model = available_llm_models[0]

    selected_llm_model = selectbox(subheader=None if choose_llm_emb_provider else "LLM",
                                   label="LLM Model",
                                   options=available_llm_models,
                                   key='LLM_MODEL',
                                   description=descriptions['LLM_MODEL'])

    return selected_llm_provider, selected_llm_model


def render_settings_tab(developer_mode):
    # Get descriptions from settings_template.py
    descriptions = get_settings_descriptions()
    provider_models = get_provider_models()
    choose_llm_emb_provider = False

    # Display settings
    col1, col2, col3 = st.columns([1/3, 1/3, 1/3])
    with col1:
        # Text Splitter Method
        selected_splitter, chunk_size, chunk_overlap = text_processing_settings_tab(descriptions)

        # Embeddings Configuration
        selected_embeddings_provider, selected_embedding_model = embeddings_settings_tab(descriptions, provider_models,
                                                                                         choose_llm_emb_provider)

    with col2:
        # Retriever Configuration
        chunk_k, search_type, score_threshold, selected_retriever, selected_splitter_child, \
            chunk_size_child, chunk_overlap_child, rerank_enabled = retrieve_settings_tab(descriptions, chunk_size)

    with col3:
        # LLM Configuration
        selected_llm_provider, selected_llm_model = llm_settings_tab(descriptions, provider_models,
                                                                     choose_llm_emb_provider=choose_llm_emb_provider)
        st.subheader("RAG Pipeline Overview")
        st.image("./images/pipeline.PNG", use_container_width=True)

    # Save Settings Button
    if st.button("Save Settings", type="primary", key="save_settings"):
        # Update settings in session state
        save_settings(selected_splitter, selected_embeddings_provider, selected_embedding_model, selected_retriever,
                      rerank_enabled, selected_llm_provider, selected_llm_model, chunk_k, chunk_size, chunk_overlap,
                      selected_splitter_child, chunk_size_child, chunk_overlap_child, search_type, score_threshold,
                      developer_mode)


def render_chat_tab(developer_mode):
    # initialize page, executed only once per session
    initialize_page()
    # Create button to exit the application. This button sets session_state['is_EXIT_clicked'] to True
    st.sidebar.button("EXIT", type="primary", on_click=click_exit_button)
    # initialize logo, executed only once per session
    initialize_logo()
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
        if confidential:
            llm_provider, llm_model, embeddings_provider, embeddings_model = \
                ut.get_relevant_models(summary=False,
                                       private=confidential)
        else:
            llm_provider = st.session_state.settings['LLM_PROVIDER']
            llm_model = st.session_state.settings['LLM_MODEL']
            embeddings_provider = st.session_state.settings['EMBEDDINGS_PROVIDER']
            embeddings_model = st.session_state.settings['EMBEDDINGS_MODEL']
            retriever_type = st.session_state.settings['RETRIEVER_TYPE']
            rerank = st.session_state.settings['RERANK']
            text_splitter_method = st.session_state.settings['TEXT_SPLITTER_METHOD']
            chunk_k = st.session_state.settings['CHUNK_K']
            chunk_size = st.session_state.settings['CHUNK_SIZE']
            chunk_overlap = st.session_state.settings['CHUNK_OVERLAP']
            text_splitter_method_child = st.session_state.settings['TEXT_SPLITTER_METHOD_CHILD']
            chunk_size_child = st.session_state.settings['CHUNK_SIZE_CHILD']
            chunk_overlap_child = st.session_state.settings['CHUNK_OVERLAP_CHILD']
            search_type = st.session_state.settings['SEARCH_TYPE']
            score_threshold = st.session_state.settings['SCORE_THRESHOLD']

        # creation of Querier object, executed only once per session
        querier = initialize_querier(my_llm_provider=llm_provider,
                                     my_llm_model=llm_model,
                                     my_embeddings_provider=embeddings_provider,
                                     my_embeddings_model=embeddings_model,
                                     my_retriever_type=retriever_type,
                                     my_rerank=rerank,
                                     my_chunk_k=chunk_k,
                                     my_search_type=search_type,
                                     my_score_threshold=score_threshold)

        # If a different folder or (set of) document(s) is chosen,
        # clear querier history and
        # set the go button session state 'is_go_clicked' to False
        if ((folder_name_selected != st.session_state['folder_selected']) or
           (document_selection != st.session_state['documents_selected'])):
            querier.clear_history()
            st.session_state['messages'] = []
            st.session_state['chat_history'] = []
            st.session_state['is_GO_clicked'] = False
            st.session_state['is_in_memory'] = ut.check_in_memory_storage(folder_path_selected)
            st.session_state['file_size_error'] = False
            st.session_state['source_image'] = {}
            st.session_state['vector_store'] = None
            st.session_state['vecdb_folder_path_selected'] = ""
            st.session_state['summary_storage'] = {}
        st.session_state['folder_selected'] = folder_name_selected
        st.session_state['documents_selected'] = document_selection

        # clear querier history if a switch in confidentiality is made
        if confidential != st.session_state['confidential']:
            querier.clear_history()
            st.session_state['messages'] = []
            st.session_state['chat_history'] = []
            st.session_state['is_GO_clicked'] = False
            st.session_state['is_in_memory'] = ut.check_in_memory_storage(folder_path_selected)
            st.session_state['file_size_error'] = False
            st.session_state['source_image'] = {}
            st.session_state['vector_store'] = None
            st.session_state['vecdb_folder_path_selected'] = ""
            st.session_state['summary_storage'] = {}
        # update session state for confidentiality
        st.session_state['confidential'] = confidential

        # create radio button group for summarization
        summary_type = st.sidebar.radio(
                                        label="Start with summary?",
                                        options=["No", "Short", "Long"],
                                        captions=["", "Quicker and shorter", "Slower but more extensive"],
                                        index=0
                                    )
        # when a different summary choice is made, set the go button session state 'is_go_clicked' to False
        if summary_type != st.session_state['is_summary_clicked']:
            st.session_state['is_GO_clicked'] = False
        st.session_state['is_summary_clicked'] = summary_type

        # create button to confirm folder selection.This button sets session_state['is_GO_clicked'] to True when clicked
        st.sidebar.button(label="GO", type="primary", on_click=click_go_button,
                          help="show the prompt bar at the bottom of the screen to take questions")

        # only start a conversation when a folder is selected and selection is confirmed with "GO" button
        if st.session_state['is_GO_clicked']:
            logger.info("GO button is clicked")
            if len(document_selection) > 0:
                # get relevant models
                # _, _, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                #                                                                      private=confidential)

                # determine name of associated vector database
                vecdb_folder_path = ut.create_vectordb_path(content_folder_path=folder_path_selected,
                                                            embeddings_provider=embeddings_provider,
                                                            embeddings_model=embeddings_model,
                                                            retriever_type=retriever_type,
                                                            text_splitter_method=text_splitter_method,
                                                            chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap,
                                                            text_splitter_method_child=text_splitter_method_child,
                                                            chunk_size_child=chunk_size_child,
                                                            chunk_overlap_child=chunk_overlap_child)
                # create or update vector database if necessary
                check_vectordb(my_querier=querier,
                               my_folder_name_selected=folder_name_selected,
                               my_folder_path_selected=folder_path_selected,
                               my_documents_selected=document_selection,
                               my_vecdb_folder_path_selected=vecdb_folder_path,
                               my_embeddings_provider=embeddings_provider,
                               my_embeddings_model=embeddings_model,
                               my_text_splitter_method=text_splitter_method,
                               my_retriever_type=retriever_type,
                               my_chunk_overlap=chunk_overlap,
                               my_chunk_size=chunk_size,
                               my_chunk_overlap_child=chunk_overlap_child,
                               my_chunk_size_child=chunk_size_child,
                               my_text_splitter_method_child=text_splitter_method_child)
                # if one of the summary creation options is chosen
                if summary_type in ["Short", "Long"]:
                    # show the summary at the top of the screen
                    create_and_show_summary(my_summary_type=summary_type,
                                            my_folder_path_selected=folder_path_selected,
                                            my_selected_documents=document_selection)
                # show button "Clear Conversation" if no file size error
                clear_messages_button = None
                if not st.session_state['file_size_error']:
                    clear_messages_button = st.button(label="Clear Conversation", key="clear")
                # if button "Clear Conversation" is clicked
                if clear_messages_button:
                    # clear all chat messages on screen and in Querier object
                    # NB: session state of "is_GO_clicked" and "folder_selected" remain unchanged
                    st.session_state['messages'] = []
                    querier.clear_history()
                    st.session_state['chat_history'] = []
                    st.session_state['source_image'] = {}
                    # st.session_state['is_GO_clicked'] = False
                    logger.info("Clear Conversation button clicked")
                # display chat messages from history
                # path is needed to show source documents after the assistant's response
                display_chat_history(folder_path_selected)

                # create chat input bar if no file size error
                prompt = None
                if not st.session_state['file_size_error']:
                    prompt = st.chat_input("Your question", key="chat_input")

                # react to user input if a question has been asked
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


def render_tab_safely(tab_name: str, render_function, developer_mode: bool = False):
    try:
        render_function(developer_mode)
    except Exception:
        st.error("An error occurred. Please try again.")
        error_details = traceback.format_exc()

        # Show error in UI (collapsible to avoid cluttering the interface)
        with st.expander("Show detailed error"):
            st.code(error_details, language='python')

        # Log to file
        logger.error(f"Error in {tab_name}:\n{error_details}")


# ### MAIN PROGRAM ####
def main():
    # set page configuration, this is the first thing that needs to be done
    import sys
    if len(sys.argv) > 1:
        print("Developer mode on")
        developer_mode = True
    else:
        developer_mode = False
    set_page_config()
    # initialize session state variables
    initialize_session_state()
    tab1, tab2 = st.tabs(["Chat", "Settings"])
    with tab1:
        render_tab_safely("render_chat_tab", render_chat_tab)
    with tab2:
        render_tab_safely("render_settings_tab", render_settings_tab, developer_mode)


if __name__ == "__main__":
    main()
