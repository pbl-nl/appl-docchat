import os
import re
from typing import Callable, List, Tuple
from loguru import logger
import langchain.text_splitter as splitter
import pandas as pd
# local imports
import settings
import utils as ut
from ingest.file_parser import FileParser


def merge_hyphenated_words(text: str) -> str:
    """
    Merge words in the text that have been split with a hyphen.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    """
    Replace single newline characters in the text with spaces.
    """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    """
    Reduce multiple newline characters in the text to a single newline.
    """
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]) -> List[Tuple[int, str]]:
    """
    Apply the cleaning functions to the text of each page.
    """
    logger.info("Cleaning text")
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def clean_pages(raw_pages: List[str]) -> List[Tuple[int, str]]:
    cleaning_functions: List = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines
    ]
    cleaned_pages = clean_text(raw_pages, cleaning_functions)
    return cleaned_pages


def create_chunks_RCT(content_folder_name: str, pages: List[str]):
    '''
    Implements the RecursiveCharacterTextSplitter of LangChain. The splitter looks for separators to split on in the 
    order in which they appear in the "separators" argumentdouble new lines (paragraph break)Once paragraphs are 
    split, then it looks at the chunk size, if a chunk is too big, then it'll split by the next separator. If the 
    chunk is still too big, then it'll move onto the next one and so forth.
    '''
    my_chunksize = 1000
    my_chunkoverlap = 200
    my_separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    my_settings = str(my_separators) + " | " + str(my_chunksize) + " | " + str(my_chunkoverlap)
    text_splitter = splitter.RecursiveCharacterTextSplitter(
        chunk_size=my_chunksize,
        chunk_overlap=my_chunkoverlap,
        separators=my_separators
        # length_function = len,
        # is_separator_regex = False
    )
    column_names = ["chunk", "RCT_settings", "RCT_text"]
    df_out = pd.DataFrame(columns=column_names)
    counter = 0
    for page in pages:
        chunks = text_splitter.split_text(page[1])
        for chunk in chunks:
            df_out.loc[counter] = [counter, my_settings, chunk]
            counter += 1
    outdir = os.path.join(settings.CHUNK_DIR, content_folder_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df_out.to_csv(path_or_buf=os.path.join(outdir, "RCTchunks.tsv"), sep="\t", index=False)


def create_chunks_NLTK(content_folder_name: str, pages: List[str]):
    my_chunksize = 1000
    my_chunkoverlap = 200
    my_separator = ["\n"]
    my_settings = str(my_separator) + " | " + str(my_chunksize) + " | " + str(my_chunkoverlap)
    text_splitter = splitter.NLTKTextSplitter(
        chunk_size=my_chunksize,
        chunk_overlap=my_chunkoverlap,
        separator="\n",
        language="dutch"
    )
    column_names = ["chunk", "NLTK_settings", "NLTK_text"]
    df_out = pd.DataFrame(columns=column_names)
    counter = 0
    for page in pages:
        chunks = text_splitter.split_text(page[1])
        for chunk in chunks:
            df_out.loc[counter] = [counter, my_settings, chunk]
            counter += 1
    outdir = os.path.join(settings.CHUNK_DIR, content_folder_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df_out.to_csv(path_or_buf=os.path.join(outdir, "NLTKchunks.tsv"), sep="\t", index=False)


def main():
    file_parser = FileParser()
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents to chunk (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, _ = ut.create_vectordb_name(content_folder_name)
    files_in_folder = os.listdir(content_folder_path)
    for file in files_in_folder:
        file_path = os.path.join(content_folder_path, file)
        raw_pages, _ = file_parser.parse_file(file_path)
        cleaned_pages = clean_pages(raw_pages)
        create_chunks_RCT(content_folder_name, cleaned_pages)
        create_chunks_NLTK(content_folder_name, cleaned_pages)


if __name__ == "__main__":
    main()
