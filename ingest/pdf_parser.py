from typing import Dict, List, Tuple
from loguru import logger
from pypdf import PdfReader
# local imports
from ingest.ingest_utils import IngestUtils


class PdfParser:
    """A parser for extracting and cleaning text from PDF documents."""

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int, text_splitter_method: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
        self.text_splitter_method = text_splitter_method

    def parse_pdf(self, pdf_file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the PDF."""
        metadata = self.extract_metadata_from_pdf(pdf_file_path)
        pages = self.extract_pages_from_pdf(pdf_file_path)
        return pages, metadata

    def extract_metadata_from_pdf(self, pdf_file_path: str) -> Dict[str, str]:
        """Extract and return the metadata from the PDF."""
        logger.info("Extracting metadata")
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)

        with open(pdf_file_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            metadata = reader.metadata
            logger.info(f"{getattr(metadata, 'title', 'no title')}")
            return {
                "title": ingestutils.getattr_or_default(metadata, 'title', '').strip(),
                "author": ingestutils.getattr_or_default(metadata, 'author', '').strip(),
                "filename": pdf_file_path.split('\\')[-1],
            }

    def extract_pages_from_pdf(self, pdf_file_path: str) -> List[Tuple[int, str]]:
        """Extract and return the text of each page from the PDF."""
        logger.info("Extracting pages")
        with open(pdf_file_path, "rb") as pdf:
            reader = PdfReader(pdf)
            return [(i + 1, p.extract_text()) for i, p in enumerate(reader.pages) if p.extract_text().strip()]
        