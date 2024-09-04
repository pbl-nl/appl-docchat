from typing import Any, Dict, List, Tuple
import numpy as np
import os
import sys
import datetime as dt
# local imports
import settings


def create_vectordb_name(content_folder_name: str,
                         retriever_type: str = None,
                         embeddings_model: str = None,
                         text_splitter_method: str = None,
                         chunk_size: int = None,
                         chunk_overlap: int = None,
                         chunk_size_child: int = None,
                         chunk_overlap_child: int = None) -> Tuple[str, str]:
    """ Creates the content folder path and vector database folder path,
    list of parameters is used to be able to run a grid search over parameters for
    comparison of evaluation results

    Parameters
    ----------
    content_folder_name : str
        name of the content folder (without the path)
    chunk_size : int, optional
        the maximum chunk size in the settings, by default None
    chunk_overlap : int, optional
        the chunk overlap in the settings, by default None

    Returns
    -------
    Tuple[str, str]
        tuple of content folder path and vector database folder path
    """
    content_folder_path = os.path.join(settings.DOC_DIR, content_folder_name)
    # vectordb_name is created from retriever_type, chunk_size, chunk_overlap, embeddings_type
    retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
    embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
    text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
    chunk_size = str(settings.CHUNK_SIZE) if chunk_size is None else str(chunk_size)
    chunk_overlap = str(settings.CHUNK_OVERLAP) if chunk_overlap is None else str(chunk_overlap)
    chunk_size_child = str(settings.CHUNK_SIZE_CHILD) if chunk_size_child is None else str(chunk_size_child)
    chunk_overlap_child = str(settings.CHUNK_OVERLAP_CHILD) if chunk_overlap_child is None else str(chunk_overlap_child)
    # vectordb_name = settings.VECDB_TYPE + "_" + chunk_size + "_" + chunk_overlap + "_" + settings.EMBEDDINGS_PROVIDER
    vectordb_name = retriever_type + "_" + embeddings_model + "_" + text_splitter_method + "_" + \
        chunk_size + "_" + chunk_overlap + "_" + chunk_size_child + "_" + chunk_overlap_child
    vectordb_folder_path = os.path.join(settings.VECDB_DIR, content_folder_name + "_" + vectordb_name)

    return content_folder_path, vectordb_folder_path


def get_relevant_files_in_folder(content_folder_path: str) -> List[str]:
    """ Creates the content folder path and vector database folder path

    Parameters
    ----------
    content_folder : str
        name of the content folder (including the path)

    Returns
    -------
    List[str]
        tuple of content folder path and vector database folder path
    """
    files_in_folder = [f for f in os.listdir(content_folder_path)
                       if ((os.path.isfile(os.path.join(content_folder_path, f))) and
                       (os.path.splitext(f)[1] in [".docx", ".html", ".md", ".pdf", ".txt"]))]

    return files_in_folder


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
