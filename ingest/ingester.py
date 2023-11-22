import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from loguru import logger
# local imports
import settings
from ingest.pdf_parser import PdfParser
from ingest.txt_parser import TxtParser
from ingest.html_parser  import HtmlParser
from ingest.word_parser import WordParser
# from ingest.content_iterator import ContentIterator
from ingest.ingest_utils import IngestUtils


class Ingester:
    '''
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
    '''
    def __init__(self, collection_name: str, content_folder: str, vectordb_folder: str, 
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, local_api_url=None,
                 file_no=None):
        load_dotenv()
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.API_URL if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no

    def ingest(self) -> None:
        '''
            Creates instances of all parsers, iterates over all files in the folder
            When parameters are read from GUI, object is initiated with parameter settings listed
        '''
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        pdf_parser = PdfParser(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        txt_parser = TxtParser(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        html_parser = HtmlParser(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        word_parser = WordParser(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)

        # determine embeddings model
        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, client=None)
            logger.info("Loaded openai embeddings")
        elif self.embeddings_provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        elif self.embeddings_provider == "local_embeddings":
            if self.local_api_url is not None:
                embeddings = OllamaEmbeddings(
                    base_url = self.local_api_url,
                    model = self.embeddings_model)
            else:
                embeddings = OllamaEmbeddings(
                    model = self.embeddings_model)
            logger.info("Loaded local embeddings: " + self.embeddings_model)

        # create empty list representing added files
        new_files = []

        if self.vecdb_type == "chromadb":
            # get all files in the folder
            files_in_folder = os.listdir(self.content_folder)
            # if the vector store already exists, get the set of ingested files from the vector store
            if os.path.exists(self.vectordb_folder):
                # define chroma vector store
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=self.vectordb_folder
                )
                logger.info(f"Vector store already exists for specified settings and folder {self.content_folder}")
                # determine the files that are added or deleted
                collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                collection_ids = [int(id) for id in collection['ids']]
                files_in_store = [metadata['filename'] for metadata in collection['metadatas']]
                files_in_store = list(set(files_in_store))
                # check if files were added or removed
                new_files = [file for file in files_in_folder if file not in files_in_store]
                files_deleted = [file for file in files_in_store if file not in files_in_folder]
                # delete all chunks from the vector store that belong to files removed from the folder
                if len(files_deleted) > 0:
                    logger.info(f"Files are deleted, so vector store for {self.content_folder} needs to be updated")
                    ids_to_delete = []
                    for idx in range(len(collection['ids'])):
                        idx_id = collection['ids'][idx]
                        idx_metadata = collection['metadatas'][idx]
                        if idx_metadata['filename'] in files_deleted:
                            ids_to_delete.append(idx_id)
                    vector_store.delete(ids_to_delete)
                    logger.info(f"Deleted files from vectorstore")
                # determine updated maximum id from collection after deletions
                collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                collection_ids = [int(id) for id in collection['ids']]
                start_id = max(collection_ids) + 1
            # else it needs to be created first
            else:
                logger.info(f"Vector store to be created for folder {self.content_folder}")
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=self.vectordb_folder
                )
                collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                # all files in the folder are to be ingested into the vector store
                new_files = [file for file in files_in_folder]
                start_id = 0

            # For all files to add to the vector store
            if len(new_files) > 0:
                logger.info(f"Files are added, so vector store for {self.content_folder} needs to be updated")
                for file in new_files:
                    file_path = os.path.join(self.content_folder, file)
                    # extract pages and metadata according to file type
                    if file.endswith(".pdf"):
                        raw_pages, metadata = pdf_parser.parse_pdf(file_path)
                    elif file.endswith(".txt") or file.endswith(".md"):
                        raw_pages, metadata = txt_parser.parse_txt(file_path)
                    elif file.endswith(".html"):
                        raw_pages, metadata = html_parser.parse_html(file_path)
                    elif file.endswith(".docx"):
                        raw_pages, metadata = word_parser.parse_word(file_path)
                    else:
                        logger.info(f"Skipping ingestion of file {file} because it has extension {file[-4:]}")
                    # convert the raw text to cleaned text chunks
                    documents = ingestutils.clean_text_to_docs(raw_pages, metadata)
                    logger.info(f"Extracted {len(documents)} chunks from {file}")    
                    # and add the chunks to the vector store 
                    vector_store.add_documents(
                        documents=documents,
                        embedding=embeddings,
                        collection_name=self.collection_name,
                        persist_directory=self.vectordb_folder,
                        ids=[str(id) for id in list(range(start_id, start_id + len(documents)))] # add id to file chunks for later identification
                    )
                    collection = vector_store.get() # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
                    collection_ids = [int(id) for id in collection['ids']]
                    start_id = max(collection_ids) + 1
                logger.info(f"Added files to vectorstore")

            # save updated vector store to disk
            vector_store.persist()
