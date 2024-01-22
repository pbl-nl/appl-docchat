from typing import Dict, List, Tuple
from loguru import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from pypdf import PdfReader
# local imports
import utils as ut


class FileParser:
    """
    A class with functionality to parse various kinds of files
    """
    def parse_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            raw_pages, metadata = self.parse_pdf(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            raw_pages, metadata = self.parse_txt(file_path)
        elif file_path.endswith(".html"):
            raw_pages, metadata = self.parse_html(file_path)
        elif file_path.endswith(".docx"):
            raw_pages, metadata = self.parse_word(file_path)
        return raw_pages, metadata

    def get_metadata(self, file_path: str, metadata_text: str):
        return {"title": ut.getattr_or_default(obj=metadata_text, attr='title', default='').strip(),
                "author": ut.getattr_or_default(obj=metadata_text, attr='author', default='').strip(),
                "filename": file_path.split('\\')[-1]
                }

    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the html file"""
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(1, raw_text)] # html files do not have multiple pages
        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = data[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        return pages, metadata

    def parse_pdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the PDF file"""
        with open(file_path, "rb") as pdf_file:
            logger.info("Extracting metadata")
            reader = PdfReader(pdf_file)
            metadata_text = reader.metadata
            logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
            metadata = self.get_metadata(file_path, metadata_text)
            logger.info("Extracting pages")
            pages = [(i + 1, p.extract_text()) for i, p in enumerate(reader.pages) if p.extract_text().strip()]
        return pages, metadata

    def parse_txt(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the text file"""
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = TextLoader(file_path)
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
        """Extract and return the pages and metadata from the word document."""
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
