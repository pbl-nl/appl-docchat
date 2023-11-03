from dotenv import load_dotenv
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain.docstore.document as docstore
from langchain.embeddings import OllamaEmbeddings
from loguru import logger
# local imports
import settings
from .pdf_parser import PdfParser
from .content_iterator import ContentIterator


class Ingester:
    # When parameters are read from settings.py, object is initiated without parameter settings
    # When parameters are read from GUI, object is initiated with parameter settings listed
    def __init__(self, collection_name: str, content_folder: str, vectordb_folder: str, 
                 embeddings_provider=None, embeddings_model=None, vecdb_type=None, chunk_size=None, chunk_overlap=None,
                 local_api_url=None, file_no=None):
        load_dotenv()
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.LOCAL_API_URL if local_api_url is None and settings.LOCAL_API_URL is not None else local_api_url
        self.file_no = file_no

    def ingest(self) -> None:
        content_iterator = ContentIterator(self.content_folder)
        # create text chunks with chosen settings of chunk size and chunk overlap
        pdf_parser = PdfParser(self.chunk_size, self.chunk_overlap, self.file_no)

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
        
        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, client=None)
            logger.info("Loaded openai embeddings")

        if self.embeddings_provider == "local_embeddings":
            if self.local_api_url is not None:
                embeddings = OllamaEmbeddings(
                    base_url = self.local_api_url,
                    model = self.embeddings_model)
            else:
                embeddings = OllamaEmbeddings(
                    model = self.embeddings_model)
            logger.info("Loaded local embeddings: " + self.embeddings_model)

        # create vector store with chosen settings of vector store type (e.g. chromadb)
        if self.vecdb_type == "chromadb":
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.vectordb_folder
            )
            vector_store.persist()
