import os
from dotenv import load_dotenv
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import langchain.docstore.document as docstore
from loguru import logger
# local imports
import settings
from .pdf_parser import PdfParser
from .txt_parser import TxtParser
from .html_parser  import HtmlParser
from .word_parser import WordParser
from .content_iterator import ContentIterator
from ingest.ingest_utils import IngestUtils


class Ingester:
    # When parameters are read from settings.py, object is initiated without parameter settings
    # When parameters are read from GUI, object is initiated with parameter settings listed
    def __init__(self, collection_name: str, content_folder: str, vectordb_folder: str, 
                 embeddings_provider=None, embeddings_model=None, vecdb_type=None, chunk_size=None, chunk_overlap=None,
                 file_no=None):
        load_dotenv()
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.file_no = file_no

    def ingest(self) -> None:
        content_iterator = ContentIterator(self.content_folder)
        # create text chunks with chosen settings of chunk size and chunk overlap
        pdf_parser = PdfParser(self.chunk_size, self.chunk_overlap, self.file_no)
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no)
        txt_parser = TxtParser(self.chunk_size, self.chunk_overlap, self.file_no)
        html_parser = HtmlParser(self.chunk_size, self.chunk_overlap, self.file_no)
        word_parser = WordParser(self.chunk_size, self.chunk_overlap, self.file_no)

        chunks: List[docstore.Document] = []
        # for each file that the content_iterator yields
        for document in content_iterator:
            # check and set document path
            if not os.path.isfile(document):
                raise FileNotFoundError(f"File not found: {document}")
            self.file_path = document

            if document.endswith(".pdf"):
                # parse pdf 
                raw_pages, metadata = pdf_parser.parse_pdf(self.file_path)

            elif document.endswith(".txt"):
                # parse txt file
                raw_pages, metadata = txt_parser.parse_txt(self.file_path)

            elif document.endswith(".md"):
                # parse md file
                raw_pages, metadata = txt_parser.parse_txt(self.file_path)

            elif document.endswith(".html"):
                # parse html file
                raw_pages, metadata = html_parser.parse_html(self.file_path)

            elif document.endswith(".docx"):
                # parse word document (as one; not separated into pages)
                raw_pages, metadata = word_parser.parse_word(self.file_path)

            else:
                logger.info(f"Cannot ingest document {document} because it has extension {document[-4:]}")

            # convert the raw text to cleaned text chunks
            document_chunks = ingestutils.clean_text_to_docs(raw_pages, metadata)
            chunks.extend(document_chunks)
            logger.info(f"Extracted {len(chunks)} chunks from {document}")

        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, client=None)
            logger.info("Loaded openai embeddings")
        
        if self.embeddings_provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)

        # create vector store with chosen settings of vector store type (e.g. chromadb)
        if self.vecdb_type == "chromadb":
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.vectordb_folder
            )
            vector_store.persist()
