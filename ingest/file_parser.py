from typing import Dict, List, Tuple
import re
from loguru import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from pypdf import PdfReader
import fitz
# from unidecode import unidecode
# import pprint
from langdetect import detect, LangDetectException
# local imports
import utils as ut
import settings_template as settings
from ingest.splitter_creator import SplitterCreator
# import ingest.pdf_analyzer as pdf_analyzer

class FileParser:
    """
    A class with functionality to parse various kinds of files
    """
    def __init__(self, text_splitter_method=None, chunk_size=None, chunk_overlap=None) -> None:
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        # define splitter
        self.splitter = SplitterCreator(self.text_splitter_method, self.chunk_size, self.chunk_overlap).get_splitter()

    def parse_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            # raw_pages, metadata = self.parse_pdf(file_path)
            raw_pages, metadata = self.parse_pymupdf(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            raw_pages, metadata = self.parse_txt(file_path)
        elif file_path.endswith(".html"):
            raw_pages, metadata = self.parse_html(file_path)
        elif file_path.endswith(".docx"):
            raw_pages, metadata = self.parse_word(file_path)
        # return raw_pages, page_info, metadata
        return raw_pages, metadata

    def get_metadata(self, file_path: str, metadata_text: str):
        """
        Extracts the following metadata from the pdf document:
        title, author(s) and full filename
        """
        return {"title": ut.getattr_or_default(obj=metadata_text, attr='title', default='').strip(),
                "author": ut.getattr_or_default(obj=metadata_text, attr='author', default='').strip(),
                "filename": file_path.split('\\')[-1]
                }

    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the html file
        """
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(1, raw_text)]  # html files do not have multiple pages
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = data[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else self._detect_language(raw_text)
        return pages, metadata

    # def parse_pdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    #     """
    #     Extract and return the pages and metadata from the PDF file
    #     """
    #     with open(file_path, "rb") as pdf_file:
    #         logger.info("Extracting metadata")
    #         reader = PdfReader(pdf_file)
    #         metadata_text = reader.metadata
    #         logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
    #         metadata = self.get_metadata(file_path, metadata_text)
    #         print('METADATA: ', metadata)
    #         logger.info("Extracting pages")
    #         for _, page in enumerate(reader.pages):
    #             page_text = unidecode(page.extract_text()).strip()
    #             print(f"{page_text}\n")
    #         pages = [(i + 1, p.extract_text()) for i, p in enumerate(reader.pages) if p.extract_text().strip()]
    #     metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else self._detect_language(pages[0][1])
    #     return pages, metadata

    def parse_pymupdf(self, file_path: str) -> Tuple[str, List[Tuple[int, str]], Dict[str, str]]:
        """
        Extracts and return the page blocks and metadata from the PDF file
        Then determines which blocks of text should be merged, depending on whether the previous block was a 
        paragraph header
        """
        logger.info("Extracting pdf metadata")
        doc = fitz.open(file_path)
        metadata_text = doc.metadata
        metadata = self.get_metadata(file_path, metadata_text)
        pages = []
        toc = doc.get_toc()
        toc_paragraph_titles = [item[1] for item in toc]
        # print(f"paragraph titles: {toc_paragraph_titles}")

        # Determine which blocks on a page represent content. We only want to feed content (including the paragraph
        # header) into the vector database
        # doc_tags = pdf_analyzer.get_doc_tags(doc)

        # for each page in pdf file
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
                        # print(f"block {block[5]}: {block_text} is a page number")

                    # block text should not represent a page header or footer containing a pipe character
                    # and some text
                    pattern_pagenr = r'^\s*(\d+)\s*\|\s*([\w\s]+)$|^\s*([\w\s]+)\s*\|\s*(\d+)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False
                        # print(f"block {block[5]}: {block_text} is a page number")

                    # # text must have "paragraph" tag
                    # if block_tag != "<p>":
                    #     block_is_valid = False

                    # block text should not represent any form of paragraph title
                    pattern_paragraph = r'^\d+(\.\d+)*\s*.+$'
                    if bool(re.match(pattern_paragraph, block_text)):
                        if not block_is_pagenr:
                            block_is_paragraph = True
                            # print(f"block {block[5]}: {block_text} is a paragraph")

                    # if current block is content
                    if block_is_valid and (not block_is_paragraph):
                        # print(f"block {block[5]} is valid and not a paragraph: {block_text} ")
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
                                print(f"added to page {i}: {prv_block_text}")
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
                print(f"added to page {i}: {prv_block_text}")

            # tabs = page.find_tables() # locate and extract any tables on page
            # print(f"{len(tabs.tables)} table found on {page}") # display number of found tables
            # if tabs.tables:  # at least one table found?
            #     pprint.pprint(tabs[0].extract())  # print content of first table
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else self._detect_language(pages[0][1])
        logger.info(f"The language detected for this document is {metadata['Language']}")
        return pages, metadata

    def parse_txt(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the text file
        """
        # load text and extract raw page
        logger.info("Extracting pages")
        loader = TextLoader(file_path=file_path, autodetect_encoding=True)
        text = loader.load()
        raw_text = text[0].page_content
        pages = [(1, raw_text)] # txt files do not have multiple pages
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = text[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else self._detect_language(raw_text)
        return pages, metadata

    def parse_word(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the word document
        """
        # load text and extract raw page
        logger.info("Extracting pages")
        loader = UnstructuredWordDocumentLoader(file_path)
        text = loader.load()
        raw_text = text[0].page_content
        pages = [(1, raw_text)] # currently not able to extract pages yet!
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = text[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else self._detect_language(raw_text)
        return pages, metadata
    
    def _detect_language(self, text: str, number_of_characters: int = 1000) -> str:
        '''
        Detects language based on the first X number of characters
        '''
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

