import os
import re
# from datetime import date
from typing import Callable, Dict, List, Tuple
import langchain.docstore.document as docstore
import langchain.text_splitter as splitter
from loguru import logger


class IngestUtils:
    """Utils for ingesting different types of documents. 
    This includes cutting text into chunks and cleaning text."""

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
    
    def getattr_or_default(self, obj, attr, default=None):
        """Get an attribute from an object, returning a default value if the attribute """
        """is not found or its value is None."""
        value = getattr(obj, attr, default)
        return value if value is not None else default

    def clean_text_to_docs(self, raw_pages, metadata) -> List[docstore.Document]:
        cleaning_functions: List = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines,
        ]

        '''
        if self.file_no > 0:
            metadata['file_no'] = self.file_no
        '''

        cleaned_text_pdf = self.clean_text(raw_pages, cleaning_functions)
        return self.text_to_docs(cleaned_text_pdf, metadata)

    def clean_text(self,
                   pages: List[Tuple[int, str]],
                   cleaning_functions: List[Callable[[str], str]]
                   ) -> List[Tuple[int, str]]:
        """Apply the cleaning functions to the text of each page."""
        logger.info("Cleaning text of each page")
        cleaned_pages = []
        for page_num, text in pages:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_pages.append((page_num, text))
        return cleaned_pages

    def merge_hyphenated_words(self, text: str) -> str:
        """Merge words in the text that have been split with a hyphen."""
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text: str) -> str:
        """Replace single newline characters in the text with spaces."""
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text: str) -> str:
        """Reduce multiple newline characters in the text to a single newline."""
        return re.sub(r"\n{2,}", "\n", text)

    def text_to_docs(self, text: List[Tuple[int, str]],
                     metadata: Dict[str, str]) -> List[docstore.Document]:
        """Split the text into chunks and return them as Documents."""
        doc_chunks: List[docstore.Document] = []

        chunk_no = 0
        for page_num, page in text:
            logger.info(f"Splitting page {page_num}")
            text_splitter = splitter.RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=self.chunk_overlap,
            )
            chunks = text_splitter.split_text(page)
            for i, chunk in enumerate(chunks):
                if self.file_no:
                    metadata_combined = {
                        "file_no": self.file_no,
                        "chunk_no": chunk_no,
                        "source": f"F{self.file_no}-{chunk_no}"
                    }
                else:
                    metadata_combined = {
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        **metadata,
                    }
                doc = docstore.Document(
                    page_content=chunk,
                    metadata=metadata_combined
                )
                doc_chunks.append(doc)
                chunk_no += 1
        return doc_chunks