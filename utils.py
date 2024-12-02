"""
The utils module contains general functionality that can be used at various places in the application
"""
from typing import Any, Dict, List, Tuple
import os
import sys
import datetime as dt
import pathlib
import numpy as np
from loguru import logger
from langdetect import detect, LangDetectException
# local imports
import settings

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


def create_vectordb_folder() -> None:
    """ Creates subfolder for storage of vector databases if not existing
    """
    if settings.VECDB_DIR not in os.listdir(pathlib.Path().resolve()):
        os.mkdir(os.path.join(pathlib.Path().resolve(), settings.VECDB_DIR))


def create_summaries_folder(content_folder_name: str) -> None:
    """ Creates subfolder for storage of summaries if not existing

    Parameters
    ----------
    content_folder_name : str
        name of the content folder (without the path)
    """
    if "summaries" not in os.listdir(os.path.join(pathlib.Path().resolve(), settings.DOC_DIR, content_folder_name)):
        os.mkdir(os.path.join(pathlib.Path().resolve(), settings.DOC_DIR, content_folder_name, "summaries"))


def create_vectordb_name(content_folder_name: str,
                         retriever_type: str = None,
                         embeddings_model: str = None,
                         text_splitter_method: str = None,
                         chunk_size: int = None,
                         chunk_overlap: int = None,
                         chunk_size_child: int = None,
                         chunk_overlap_child: int = None) -> Tuple[str, str]:
    """ Creates the content folder path and vector database folder path

    Parameters
    ----------
    content_folder_name : str
        name of the content folder (without the path)
    retriever_type : str, optional
        name of the retriever type, by default None
    embeddings_model : str, optional
        name of the embeddings_model, by default None
    text_splitter_method : str, optional
        name of the text splitter method, by default None
    chunk_size : int, optional
        the maximum chunk size, by default None
    chunk_overlap : int, optional
        the chunk overlap, by default None
    chunk_size_child : int, optional
        the maximum chunk size of child chunks, by default None
    chunk_overlap_child : int, optional
        the chunk overlap of child chunks, by default None

    Returns
    -------
    Tuple[str, str]
        tuple of content folder path and vector database folder path
    """
    content_folder_path = os.path.join(settings.DOC_DIR, content_folder_name)
    retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
    embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
    text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
    chunk_size = str(settings.CHUNK_SIZE) if chunk_size is None else str(chunk_size)
    chunk_overlap = str(settings.CHUNK_OVERLAP) if chunk_overlap is None else str(chunk_overlap)
    chunk_size_child = str(settings.CHUNK_SIZE_CHILD) if chunk_size_child is None else str(chunk_size_child)
    chunk_overlap_child = str(settings.CHUNK_OVERLAP_CHILD) \
        if chunk_overlap_child is None else str(chunk_overlap_child)
    # vectordb_name is created from retriever_type, embeddings_model, text_splitter_method and
    # parent and child chunk_size and chunk_overlap
    vectordb_name = content_folder_name + "_" + retriever_type + "_" + embeddings_model + "_" + \
        text_splitter_method + "_" + chunk_size + "_" + chunk_overlap + "_" + chunk_size_child + "_" + \
        chunk_overlap_child

    vectordb_folder_path = os.path.join(pathlib.Path().resolve(), settings.VECDB_DIR, vectordb_name)

    return content_folder_path, vectordb_folder_path


def is_relevant_file(content_folder_path: str, my_file: str) -> bool:
    """
    decides whether or not a file is a relevant file

    Parameters
    ----------
    content_folder_path : str
        name of the content folder (including the path)
    my_file: str
        name of the file

    Returns
    -------
    bool
        True if file is relevant, otherwise False
    """
    relevant = ((os.path.isfile(os.path.join(content_folder_path, my_file))) and
                (os.path.splitext(my_file)[1] in [".docx", ".html", ".md", ".pdf", ".txt"]))
    if not relevant:
        logger.info(f"Skipping ingestion of file {my_file} because it has extension {os.path.splitext(my_file)[1]}")

    return relevant


def get_relevant_files_in_folder(content_folder_path: str) -> List[str]:
    """ Gets a list of relevant files from a given content folder path

    Parameters
    ----------
    content_folder_path : str
        name of the content folder (including the path)

    Returns
    -------
    List[str]
        list of files, without path
    """
    return [f for f in os.listdir(content_folder_path) if is_relevant_file(content_folder_path, f)]


def exit_program() -> None:
    """ Exits the Python process
    """
    print("Exiting the program...")
    sys.exit(0)


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


def get_relevant_models(private: bool) -> Tuple[str, str, str, str]:
    if private:
        return settings.PRIVATE_LLM_PROVIDER, settings.PRIVATE_LLM_MODEL, \
               settings.PRIVATE_EMBEDDINGS_PROVIDER, settings.PRIVATE_EMBEDDINGS_MODEL
    else:
        return settings.LLM_PROVIDER, settings.LLM_MODEL, \
               settings.EMBEDDINGS_PROVIDER, settings.EMBEDDINGS_MODEL
