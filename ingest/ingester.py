from dotenv import load_dotenv
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain.docstore.document as docstore
from loguru import logger
# local imports
from .pdf_parser import PdfParser
from .content_iterator import ContentIterator


class Ingester:

    def __init__(self, input_folder: str, content_folder: str, vectordb_folder: str, embeddings_type: str, vectordb_type: str, chunk_size: int, chunk_overlap: int):
        load_dotenv()
        self.input_folder = input_folder
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.vectordb_type = vectordb_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_type = embeddings_type

    def ingest(self) -> None:
        content_iterator = ContentIterator(self.content_folder)
        pdf_parser = PdfParser(self.chunk_size, self.chunk_overlap)

        chunks: List[docstore.Document] = []
        # for each pdf file that the content_iterator yields
        for document in content_iterator:
            if document.endswith(".pdf"):
                # convert the pdf text to cleaned text chunks
                pdf_parser.set_pdf_file_path(document)
                document_chunks = pdf_parser.clean_text_to_docs()
                chunks.extend(document_chunks)
                logger.info(f"Extracted {len(chunks)} chunks from {document}")
            else:
                logger.info(f"Cannot ingest document {document} because it has extension {document[-4:]}")
        
        if self.embeddings_type == "openai":
            embeddings = OpenAIEmbeddings(client=None)
            logger.info("Loaded openai embeddings")

        if self.vectordb_type == "chromadb":
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.input_folder,
                persist_directory=self.vectordb_folder
            )
            vector_store.persist()
            logger.info(f"Created Chroma vector store in folder {self.vectordb_folder}")
