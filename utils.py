from typing import Any, Dict, Tuple
import numpy as np
import os
import sys
import datetime as dt
# local imports
import settings


def create_vectordb_name(content_folder_name: str,
                         chunk_size: int = None,
                         chunk_overlap: int = None) -> Tuple[str, str]:
    """ Creates the content folder path and vector database folder path

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
    # vectordb_name is created from vecdb_type, chunk_size, chunk_overlap, embeddings_type
    chunk_size = str(settings.CHUNK_SIZE) if chunk_size is None else str(chunk_size)
    chunk_overlap = str(settings.CHUNK_OVERLAP) if chunk_overlap is None else str(chunk_overlap)
    vectordb_name = settings.VECDB_TYPE + "_" + chunk_size + "_" + chunk_overlap + "_" + settings.EMBEDDINGS_PROVIDER
    vectordb_folder_path = os.path.join(settings.VECDB_DIR, content_folder_name + "_" + vectordb_name)

    return content_folder_path, vectordb_folder_path


def exit_program():
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
        name of the settings file

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


def get_timestamp():
    """ returns the current time as a string, used for logging

    Returns
    -------
    str
        string timestamp of current time
    """

    return str(dt.datetime.now())


def euclidean_distance(a, b):
    """
    Calculation of euclidian distance between a and b
    """

    return np.sqrt(np.sum((a - b) ** 2))


def get_relevant_files_from_folder(content_folder):
    files_in_folder = [f for f in os.listdir(content_folder)
                       if os.path.isfile(os.path.join(content_folder, f))]
    relevant_files_in_folder = []
    for file in files_in_folder:
        # file_path = os.path.join(self.content_folder, file)
        _, file_extension = os.path.splitext(file)
        if file_extension in [".docx", ".html", ".md", ".pdf", ".txt"]:
            relevant_files_in_folder.append(file)

    return relevant_files_in_folder
