"""
The utils module contains general functionality that can be used at various places in the application
"""
# imports
from typing import Any, Dict, List, Tuple
import os
import sys
import time
import datetime as dt
import numpy as np
from loguru import logger
from langdetect import detect, LangDetectException
import psutil
import keyboard
# local imports
import settings

VALID_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".html",
    ".md",
    ".txt"
]

LANGUAGE_MAP = {
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'it': 'italian',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tr': 'turkish'
}  # languages supported by nltk


def create_vectordb_folder(my_folder_path_selected: str) -> None:
    """
    Creates subfolder for storage of vector databases if not existing

    Parameters
    ----------
    my_folder_path_selected : str
        the selected document folder path

    """
    if "vector_stores" not in os.listdir(my_folder_path_selected):
        os.mkdir(os.path.join(my_folder_path_selected, "vector_stores"))


def create_summaries_folder(my_folder_path_selected: str) -> None:
    """
    Creates subfolder for storage of summaries if not existing

    Parameters
    ----------
    my_folder_path_selected : str
        the selected document folder path
    """
    if "summaries" not in os.listdir(my_folder_path_selected):
        os.mkdir(os.path.join(my_folder_path_selected, "summaries"))


def create_vectordb_path(content_folder_path: str,
                         retriever_type: str = None,
                         embeddings_provider: str = None,
                         embeddings_model: str = None,
                         text_splitter_method: str = None,
                         chunk_size: int = None,
                         chunk_overlap: int = None,
                         text_splitter_method_child: str = None,
                         chunk_size_child: int = None,
                         chunk_overlap_child: int = None) -> str:
    """
    Creates the full path for the vectorstore

    Parameters
    ----------
    content_folder_path : str
        name of the content folder (including the path)
    retriever_type : str, optional
        name of the retriever type, by default None
    embeddings_provider : str, optional
        name of the embeddings provider, by default None
    embeddings_model : str, optional
        name of the embeddings model, by default None
    text_splitter_method : str, optional
        name of the text splitter method, by default None
    chunk_size : int, optional
        the maximum chunk size, by default None
    chunk_overlap : int, optional
        the chunk overlap, by default None
    text_splitter_method_child : str, optional
        name of the text splitter method used for child chunks, by default None
    chunk_size_child : int, optional
        the maximum chunk size of child chunks, by default None
    chunk_overlap_child : int, optional
        the chunk overlap of child chunks, by default None

    Returns
    -------
    str
        vectorstore folder path
    """
    retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
    embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
    embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
    text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
    chunk_size = str(settings.CHUNK_SIZE) if chunk_size is None else str(chunk_size)
    chunk_overlap = str(settings.CHUNK_OVERLAP) if chunk_overlap is None else str(chunk_overlap)
    text_splitter_method_child = settings.TEXT_SPLITTER_METHOD_CHILD if text_splitter_method_child is None else \
        text_splitter_method_child
    chunk_size_child = str(settings.CHUNK_SIZE_CHILD) if chunk_size_child is None else str(chunk_size_child)
    chunk_overlap_child = str(settings.CHUNK_OVERLAP_CHILD) \
        if chunk_overlap_child is None else str(chunk_overlap_child)
    # vectordb_name is created from retriever_type, embeddings_provider, embeddings_model, and
    # parent and child text_splitter_method, chunk_size and chunk_overlap
    vectordb_name = retriever_type + "_" + embeddings_provider + "_" + embeddings_model + "_" + \
        text_splitter_method + "_" + chunk_size + "_" + chunk_overlap + "_" + text_splitter_method_child + "_" +\
        chunk_size_child + "_" + chunk_overlap_child

    vectordb_folder_path = os.path.join(content_folder_path, "vector_stores", vectordb_name)

    return vectordb_folder_path


def is_relevant_file(content_folder_path: str, document_selection: List[str], my_file: str) -> bool:
    """
    Decides whether or not a file is a relevant file

    Parameters
    ----------
    content_folder_path : str
        name of the content folder (including the path)
    document_selection: List[str]
        list of documents that have been selected
    my_file: str
        name of the file

    Returns
    -------
    bool
        True if file is relevant, otherwise False
    """
    relevant = False
    if ((document_selection is None) or (document_selection == ["All"])):
        relevant = ((os.path.isfile(os.path.join(content_folder_path, my_file))) and
                    (os.path.splitext(my_file)[1] in VALID_EXTENSIONS) and
                    (not my_file.startswith("~")))
    else:
        relevant = ((os.path.isfile(os.path.join(content_folder_path, my_file))) and
                    (os.path.splitext(my_file)[1] in VALID_EXTENSIONS) and
                    (not my_file.startswith("~")) and
                    (my_file in document_selection))

    return relevant


def get_relevant_files_in_folder(content_folder_path: str, document_selection: List[str] = None) -> List[str]:
    """
    Gets a list of relevant files from a given content folder path

    Parameters
    ----------
    content_folder_path : str
        name of the content folder (including the path)
    document_selection: List[str]
        list of documents that have been selected

    Returns
    -------
    List[str]
        list of files, without path
    """
    all_files = os.listdir(content_folder_path)
    relevant_files = []
    for file in all_files:
        if is_relevant_file(content_folder_path=content_folder_path,
                            document_selection=document_selection,
                            my_file=file):
            logger.info(f"file {file} is found relevant for ingestion")
            relevant_files.append(file)

    return relevant_files


def exit_program() -> None:
    """
    Exits the Python process
    """
    logger.info("Exiting the program...")
    sys.exit(0)


def exit_ui() -> None:
    """
    Exits the User Interface process.
    First, the last tab in the browser is closed
    Then the python process is stopped
    From: https://discuss.streamlit.io/t/close-streamlit-app-with-button-click/35132
    """
    # Give a bit of delay for user experience
    time.sleep(1)
    # Close streamlit browser tab
    keyboard.press_and_release('ctrl+w')
    logger.info("Closing the application")
    # Terminate streamlit python process
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()


def getattr_or_default(obj: Any,
                       attr, default: Any = None) -> Any | None:
    """ Get an attribute from an object, returning a default value if the attribute
        is not found or its value is None

    Parameters
    ----------
    obj : Any
        object from which to obtain the attribute
    attr : str
        name of the attribute
    default : Any, optional
        default argument, by default None

    Returns
    -------
    Any or None
        value of the attribute if not None else the default value, which can be None
    """
    value = getattr(obj, attr, default)

    return value if value is not None else default


def get_settings_as_dictionary(file_name: str) -> Dict[str, Any]:
    """ Turns the parameters read from the settings file into a dictionary

    Parameters
    ----------
    file_name : str
        name of the settings file path

    Returns
    -------
    Dict[str, Any]
        dictionary with parameter names as keys, parameter values as values
    """
    # Initialize an empty dictionary to store the variables and their values
    variables_dict = {}
    # Open and read the file
    with open(file=file_name, mode='r', encoding="utf-8") as file:
        lines = file.readlines()
    start_reading = False
    # Process each line in the file
    for line in lines:
        # start reading below the line with # #########
        if line.startswith("# #########"):
            start_reading = True
        # ignore comment lines
        if start_reading and not line.startswith("#"):
            # Remove leading and trailing whitespace and split the line by '='
            parts = line.strip().split('=')
            # Check if the line can be split into two parts
            if len(parts) == 2:
                # Extract the variable name and value
                variable_name = parts[0].strip()
                variable_value = parts[1].strip()
                # exclude embedding map and llm map
                if not variable_value.startswith("{"):
                    # Use exec() to assign the value to the variable name
                    exec(f'{variable_name} = {variable_value}')
                    # Add the variable and its value to the dictionary
                    variables_dict[variable_name] = eval(variable_name)

    return variables_dict


def get_timestamp() -> str:
    """ returns the current time as a string, used for logging

    Returns
    -------
    str
        string timestamp of current time
    """
    return str(dt.datetime.now())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    _summary_

    Parameters
    ----------
    a : np.ndarray
        first array of numbers
    b : np.ndarray
        second array of numbers

    Returns
    -------
    np.float64
        the calculated cosine similarity between the two arrays of numbers
    """
    cos_sim = np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

    return cos_sim


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Calculation of euclidean distance between a and b

    Parameters
    ----------
    a : np.ndarray
        first array of numbers
    b : np.ndarray
        second array of numbers

    Returns
    -------
    np.float64
        the calculated euclidean distance between the two arrays of numbers
    """
    eucl_dist = np.sqrt(np.sum((a - b) ** 2))

    return eucl_dist


def detect_language(text: str, number_of_characters: int = 1000) -> str:
    """
    Detects language based on the first X number of characters
    """
    text_snippet = text[:number_of_characters] if len(text) > number_of_characters else text

    if not text_snippet.strip():
        # Handle the case where the text snippet is empty or only contains whitespace
        return 'unknown'
    try:
        return detect(text_snippet)
    except LangDetectException as e:
        if 'No features in text' in str(e):
            # Handle the specific error where no features are found in the text
            return 'unknown'


def get_relevant_models(summary: bool, private: bool) -> Tuple[str, str, str, str]:
    """
    Gets the appropriate embeddings provider and model and llm provider and model
    based on whether a summary is wanted and/or whether the documents involved are private
    Parameters
    ----------
    summary : bool
        indicator for summary as purpose
    private : bool
        indicator for private document
    Returns
    -------
    Tuple[str, str, str, str]
        tuple of embedding provider, embedding model, llm provider and llm model
    """
    if private:
        return settings.PRIVATE_LLM_PROVIDER, settings.PRIVATE_LLM_MODEL, \
               settings.PRIVATE_EMBEDDINGS_PROVIDER, settings.PRIVATE_EMBEDDINGS_MODEL
    else:
        if summary:
            return settings.SUMMARY_LLM_PROVIDER, settings.SUMMARY_LLM_MODEL, \
                   settings.SUMMARY_EMBEDDINGS_PROVIDER, settings.SUMMARY_EMBEDDINGS_MODEL
        else:
            return settings.LLM_PROVIDER, settings.LLM_MODEL, \
                   settings.EMBEDDINGS_PROVIDER, settings.EMBEDDINGS_MODEL


def answer_idontknow(language: str) -> str:
    """
    Returns the answer "I don't know" in the correct language

    Parameters
    ----------
    language : str
        language of the answer

    Returns
    -------
    str
        the answer "I don't know" in the correct language
    """
    if language == 'nl':
        result = "Ik weet het niet omdat er geen relevante context is die het antwoord bevat"
    elif language == 'de':
        result = "Ich weiß es nicht, weil es keinen relevanten Kontext gibt, der die Antwort enthält"
    elif language == 'fr':
        result = "Je ne sais pas car il n'y a pas de contexte pertinent contenant la réponse"
    else:
        result = "I don't know because there is no relevant context containing the answer"

    return result


def check_size(list_of_files, content_folder):
    """
    Checks the size of the list of files and returns the size of files in memory

    Parameters
    ----------
    list_of_files : List[str]
        list of files
    content_folder : str

    Returns
    -------
    float
        size of files in MB
    """
    size = 0
    for file in list_of_files:
        size += os.path.getsize(os.path.join(content_folder, file))
    size = size / 1024 / 1024  # convert to MB
    return size
