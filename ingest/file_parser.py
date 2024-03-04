from typing import Dict, List, Tuple
from loguru import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from pypdf import PdfReader
import fitz
from unidecode import unidecode
import re
import pprint
# local imports
import utils as ut
import settings_template as settings
from ingest.splitter import Splitter
import ingest.pdf_analyzer as pdf_analyzer

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
        self.splitter = Splitter(self.text_splitter_method, self.chunk_size, self.chunk_overlap).get_splitter()

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
        return pages, metadata

    def parse_pdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the PDF file
        """
        with open(file_path, "rb") as pdf_file:
            logger.info("Extracting metadata")
            reader = PdfReader(pdf_file)
            metadata_text = reader.metadata
            logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
            metadata = self.get_metadata(file_path, metadata_text)
            logger.info("Extracting pages")
            for _, page in enumerate(reader.pages):
                page_text = unidecode(page.extract_text()).strip()
                print(f"{page_text}\n")
            pages = [(i + 1, p.extract_text()) for i, p in enumerate(reader.pages) if p.extract_text().strip()]
        return pages, metadata

    # def is_string_contained(self, string_to_find: str, list_of_strings: List[str]) -> bool:
    #     return any(string_to_find in string for string in list_of_strings)

    def parse_pymupdf(self, file_path: str) -> Tuple[str, List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the page blocks and metadata from the PDF file
        Then determine which blocks of text should be merged, depending on whether the previous block was a header
        """
        logger.info("Extracting pdf metadata")
        doc = fitz.open(file_path)
        metadata_text = doc.metadata
        print(f"metadata = {metadata_text}")
        metadata = self.get_metadata(file_path, metadata_text)

        pages = []
        toc = doc.get_toc()
        toc_paragraph_titles = [item[1] for item in toc]
        print(f"paragraph titles: {toc_paragraph_titles}")

        # NB: paragraph text in toc is without decimal point after paragraph number, but in text it is including decimal point
        # Change this to be able to filter blocks when they are in toc
        print(f"toc = {toc}")

        # First determine, based on font size and font style and toc info, which blocks represents headers
        # and which blocks represent content. We only want to feed content into the vector database
        doc_tags = pdf_analyzer.get_doc_tags(doc)

        # for each page in pdf file
        for i, page in enumerate(doc.pages()):
            # if (page.number > 95) and (page.number) < 100:
            # obtain the blocks
            previous_block_id = -1
            page_blocks = page.get_text("blocks")
            print(f"page {page} contains {len(page_blocks)} blocks")
            # for each block
            for block in page_blocks:
                # only take the text
                if block[6] == 0:
                    block_is_valid = True
                    block_id = block[5]
                    print(f"block id = {block_id}")
                    if block_id != previous_block_id:
                        block_tag = pdf_analyzer.get_block_tag(doc_tags, i, block_id)
                        print(f"page = {i}, block_id = {block_id}, tag = {block_tag}")
                        # filter blocks
                        # if block_tag != "<p>":
                        #     block_is_valid = False
                        block_text = unidecode(block[4]).strip()
                        # # If the block text represents a paragraph, strip the paragraph number from the text
                        # pattern = r'^\d+(\.\d+)*\s+'
                        # block_text = re.sub(pattern, '', block_text)
                        # print(f"page {i} block size = {len(block_text)}: {block_text}\n")
                        if block_is_valid:
                            pages.append((i, block_text))
                        else:
                            print(f"block_text {block_text} is a paragraph title")
                    previous_block_id = block[5]
                # filter page numbers
                # filter paragraph titles
                # merge blocks when applicable
                # split merged blocks if they are too large
            # tabs = page.find_tables() # locate and extract any tables on page
            # print(f"{len(tabs.tables)} table found on {page}") # display number of found tables
            # if tabs.tables:  # at least one table found?
            #     pprint.pprint(tabs[0].extract())  # print content of first table
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
        return pages, metadata
