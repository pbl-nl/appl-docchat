import os
import re
# from datetime import date
from typing import Callable, Dict, List, Tuple
import langchain.docstore.document as docstore
import langchain.text_splitter as splitter
from loguru import logger
from pypdf import PdfReader
import settings
# local imports
from ingest.ingest_utils import IngestUtils


class PdfParser:
    """A parser for extracting text from PDF documents."""

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int, text_splitter_method: None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
        self.text_splitter_method = text_splitter_method

    def set_pdf_file_path(self, pdf_file_path: str):
        """Set the path to the PDF file."""
        if not os.path.isfile(pdf_file_path):
            raise FileNotFoundError(f"File not found: {pdf_file_path}")
        self.pdf_file_path = pdf_file_path

    def parse_pdf(self) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the PDF."""
        metadata = self.extract_metadata_from_pdf()
        pages = self.extract_pages_from_pdf()
        return pages, metadata

    def extract_metadata_from_pdf(self) -> Dict[str, str]:
        """Extract and return the metadata from the PDF."""
        logger.info("Extracting metadata")
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)

        with open(self.pdf_file_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            metadata = reader.metadata
            logger.info(f"{getattr(metadata, 'title', 'no title')}")
            # default_date = date(1900, 1, 1)
            return {
                # TODO add metadata title + pdf filename

                "title": ingestutils.getattr_or_default(metadata, 'title', '').strip(),
                "author": ingestutils.getattr_or_default(metadata, 'author', '').strip(),
                "document_name": self.pdf_file_path.split('\\')[-1],
                # "creation_date": ingestutils.getattr_or_default(metadata,
                #                                                 'creation_date',
                #                                                  default_date).strftime('%Y-%m-%d'),
            }

    def extract_pages_from_pdf(self) -> List[Tuple[int, str]]:
        """Extract and return the text of each page from the PDF."""
        logger.info("Extracting pages")
        with open(self.pdf_file_path, "rb") as pdf:
            reader = PdfReader(pdf)
            # numpages = len(reader.pages)
            return [(i + 1, p.extract_text())
                    for i, p in enumerate(reader.pages) if p.extract_text().strip()]