"""
This module represents a class with functionality to parse various kinds of files.
Raw text is extracted from the documents and metadata added.
PDF files are parsed with pymupdf, Word files are first converted to pdf files and then
parsed. This is to be able to show the sources in the converted pdf in the Streamlit UI.

Preparation for ingestion of text into vectorstore
"""
# imports
from typing import Dict, List, Tuple
import os
import re
from loguru import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
import fitz
from docx2pdf import convert
import pymupdf4llm
# local imports
import utils as ut


class FileParser:
    """
    A class with functionality to parse various kinds of files
    Preparation for ingestion of text into vectorstore
    """
    def __init__(self) -> None:
        pass

    def parse_file(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Calls function for parsing the file depending on the file extension

        Parameters
        ----------
        file_path : str
            the filepath of the file to parse

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[str, str]]
            tuple of pages (list of tuples of pagenumbers and page texts) and metadata (dictionary)
        """
        tables = []
        if file_path.endswith(".pdf"):
            raw_pages, metadata, tables = self.parse_pymupdf(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            raw_pages, metadata = self.parse_txt(file_path)
        elif file_path.endswith(".html"):
            raw_pages, metadata = self.parse_html(file_path)
        elif file_path.endswith(".docx"):
            raw_pages, metadata = self.parse_word(file_path)

        # return raw text from pages and metadata
        return raw_pages, metadata, tables

    def get_metadata(self, file_path: str, doc_metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Extracts the following metadata from the pdf document:
        title, author(s) and full filename
        For CLO, indicator_url and indicator_closed are added, they will have no effect for other documents

        Parameters
        ----------
        file_path : str
            the filepath of the file to parse
        doc_metadata : Dict[str, str]
            metadata of the document

        Returns
        -------
        Dict[str, str]
            extended metadata of the document
        """
        return {"title": doc_metadata.get('title', '').strip(),
                "author": doc_metadata.get('author', '').strip(),
                # created specifically for CLO
                "indicator_url": doc_metadata.get('keywords', '').strip(),
                "indicator_closed": doc_metadata.get('trapped', '').strip(),
                # add filename to metadata
                "filename": file_path.split('\\')[-1]
                }

    def convert_docx_to_pdf(self, docx_path: str) -> str:
        """
        converts a Word file (.docx) to a pdf file and stores the pdf file in subfolder "conversions"

        Parameters
        ----------
        docx_path : str
            path of Word file to parse

        Returns
        -------
        str
            path of output pdf file
        """
        folder, file = os.path.split(docx_path)
        pdf_path = os.path.join(folder, "conversions", file + '.pdf')
        if not os.path.exists(os.path.join(folder, 'conversions')):
            os.mkdir(os.path.join(folder, 'conversions'))
        convert(input_path=docx_path, output_path=pdf_path, keep_active=False)

        return pdf_path

    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the html file

        Parameters
        ----------
        file_path : str
            the filepath of the html file to parse

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[str, str]]
            tuple of pages (list of tuples of pagenumbers and page texts) and metadata (dictionary)
        """
        # load text and extract raw page
        logger.info("Extracting text from html file")
        loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(1, raw_text)]  # html files do not have multiple pages
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = data[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else \
            ut.detect_language(raw_text)
        logger.info(f"The language detected for this document is {metadata['Language']}")
        metadata["last_change_time"] = os.stat(file_path).st_mtime

        return pages, metadata

    def parse_pymupdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extracts and return the page blocks and metadata from the PDF file
        Then determines which blocks of text should be merged, depending on whether the previous block was a
        paragraph header

        Parameters
        ----------
        file_path : str
            the filepath of the pdf file to parse

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[str, str]]
            tuple of pages (list of tuples of pagenumbers and page texts) and metadata (dictionary)
        """
        logger.info("Extracting pdf metadata")
        doc = fitz.open(file_path)
        # when pdf is a conversion, store the original filename in the vectorstore
        if 'conversions' in file_path:
            file_path = file_path.replace('conversions\\', '').split('.pdf')[0]
        metadata = self.get_metadata(file_path, doc.metadata)
        pages = []
        page_with_max_text = -1
        max_page_text_length = -1

        # for each page in pdf file
        logger.info("Extracting text from pdf file")
        for i, page in enumerate(doc.pages()):
            first_block_of_page = True
            prv_block_text = ""
            prv_block_is_valid = True
            prv_block_is_paragraph = False
            # obtain the blocks
            blocks = page.get_text("blocks")

            # for each block
            for block in blocks:
                # only consider text blocks
                # if block["type"] == 0:
                if block[6] == 0:
                    block_is_valid = True
                    block_is_pagenr = False
                    block_is_paragraph = False
                    # block_tag = pdf_analyzer.get_block_tag(doc_tags, i, block_id)
                    # block_text = pdf_analyzer.get_block_text(doc_tags, i, block_id)
                    block_text = block[4]

                    # block text should not represent a page header or footer
                    pattern_pagenr = r'^\s*(\d+)([.\s]*)$|^\s*(\d+)([.\s]*)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False

                    # block text should not represent a page header or footer containing a pipe character
                    # and some text
                    pattern_pagenr = r'^\s*(\d+)\s*\|\s*([\w\s]+)$|^\s*([\w\s]+)\s*\|\s*(\d+)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False

                    # block text should not represent any form of paragraph title
                    pattern_paragraph = r'^\d+(\.\d+)*\s*.+$'
                    if bool(re.match(pattern_paragraph, block_text)):
                        if not block_is_pagenr:
                            block_is_paragraph = True

                    # if current block is content
                    if block_is_valid and (not block_is_paragraph):
                        # and the previous block was a paragraph
                        if prv_block_is_paragraph:
                            # extend the paragraph block text with a newline character and the current block text
                            block_text = prv_block_text + "\n" + block_text
                        # but if the previous block was a content block
                        else:
                            if prv_block_is_valid and block_is_valid:
                                # extend the content block text with a whitespace character and the current block text
                                block_text = prv_block_text + " " + block_text
                        # in both cases, set the previous block text to the current block text
                        prv_block_text = block_text
                    # else if current block text is not content
                    else:
                        # and the current block is not the very first block of the page
                        if not first_block_of_page:
                            # if previous block was content
                            if prv_block_is_valid and (not prv_block_is_paragraph):
                                # add text of previous block to pages together with page number
                                pages.append((i, prv_block_text))
                                # and empty the previous block text
                                prv_block_text = ""
                            # if previous block was not relevant
                            else:
                                # just set the set the previous block text to the current block text
                                prv_block_text = block_text

                    # set previous block validity indicators to current block validity indicators
                    prv_block_is_valid = block_is_valid
                    # prv_block_is_pagenr = block_is_pagenr
                    prv_block_is_paragraph = block_is_paragraph
                    prv_block_text = block_text

                    # set first_block_of_page to False
                    first_block_of_page = False

            # end of page:
            # if previous block was content
            if prv_block_is_valid and (not prv_block_is_paragraph):
                # add text of previous block to pages together with page number
                pages.append((i, prv_block_text))

            # store pagenr with maximum amount of characters for language detection of document
            page_text_length = len(pages[i][1])
            if page_text_length > max_page_text_length:
                page_with_max_text = i
                max_page_text_length = page_text_length

            # tabs = page.find_tables() # locate and extract any tables on page
            # print(f"{len(tabs.tables)} table found on {page}") # display number of found tables
            # if tabs.tables:  # at least one table found?
            #     pprint.pprint(tabs[0].extract())  # print content of first table

        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else \
            ut.detect_language(pages[page_with_max_text][1])
        logger.info(f"The language detected for this document is {metadata['Language']}")
        metadata["last_change_time"] = os.stat(file_path).st_mtime
        tables = self.extract_tables(file_path)

        return pages, metadata, tables

    def extract_tables(self, file_path, show_progress=False):
        """
        first chunks each page of the pdf file into markdown and then extracts tables from the markdown text.
        md_text_list contains image, graphics, metadata, table positions, and text of the pdf file in markdown format.
        Might be helpful in text parsing.

        Parameters
        ----------
        file_path : str
            path to the pdf file
        show_progress : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            list of tuples, each tuple contains the page number and the table in markdown format
        """
        md_text_list = pymupdf4llm.to_markdown(file_path, page_chunks=True, show_progress=show_progress)
        # Regular expression to detect Markdown tables
        table_pattern = re.compile(r"(\|.*?\|\n\|[-|]+\|\n(?:\|.*?\|\n)+)")

        # Process each page to find Markdown tables
        table_pages = []
        for i, md_text in enumerate(md_text_list):
            text = md_text['text']
            tables = table_pattern.findall(text)
            if tables:
                table_pages.extend([(i, table) for table in tables])

        return table_pages

    def parse_txt(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the text file

        Parameters
        ----------
        file_path : str
            the filepath of the txt file to parse

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[str, str]]
            tuple of pages (list of tuples of pagenumbers and page texts) and metadata (dictionary)
        """
        # load text and extract raw page
        logger.info("Extracting text from txt file")
        loader = TextLoader(file_path=file_path, autodetect_encoding=True)
        text = loader.load()
        raw_text = text[0].page_content
        # txt files do not have multiple pages
        pages = [(1, raw_text)]
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = text[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else \
            ut.detect_language(raw_text)
        logger.info(f"The language detected for this document is {metadata['Language']}")
        metadata["last_change_time"] = os.stat(file_path).st_mtime

        return pages, metadata

    def parse_word(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        First, the word file is converted to a pdf file
        Then the pages and metadata are extracted and returned from the pdf document

        Parameters
        ----------
        file_path : str
            the filepath of the docx file to parse

        Returns
        -------
        Tuple[List[Tuple[int, str]], Dict[str, str]]
            tuple of pages (list of tuples of pagenumbers and page texts) and metadata (dictionary)
        """
        # convert docx to pdf
        path_to_pdf = self.convert_docx_to_pdf(file_path)
        pages, metadata, _ = self.parse_pymupdf(path_to_pdf)

        return pages, metadata
