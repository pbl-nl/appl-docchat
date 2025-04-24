"""
Ingester class
Creates Ingester object
Also parses files, chunks the files and stores the chunks in vector store
When instantiating without parameters, attributes get values from settings.py
"""
# imports
import os
import re
from typing import Callable, Dict, List, Tuple
import time
from loguru import logger
import langchain.docstore.document as docstore
from dotenv import load_dotenv
import tiktoken
# local imports
import settings
import utils as ut
from ingest.file_parser import FileParser
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator
from ingest.splitter_creator import SplitterCreator


class Ingester:
    """
    Create Ingester object
    When instantiating without parameters, attributes get values from settings.py
    """
    def __init__(self, collection_name: str, content_folder: str, vecdb_folder: str,
                 document_selection: List[str] = None, embeddings_provider: str = None, embeddings_model: str = None,
                 retriever_type: str = None, text_splitter_method: str = None,
                 text_splitter_method_child: str = None, chunk_size: int = None, chunk_size_child: int = None,
                 chunk_overlap: int = None, chunk_overlap_child: int = None) -> None:
        load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vecdb_folder = vecdb_folder
        self.document_selection = document_selection
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.text_splitter_method_child = settings.TEXT_SPLITTER_METHOD_CHILD \
            if text_splitter_method_child is None else text_splitter_method_child
        self.chunk_size_child = settings.CHUNK_SIZE_CHILD if chunk_size_child is None else chunk_size_child
        self.chunk_overlap_child = settings.CHUNK_OVERLAP_CHILD if chunk_overlap_child is None else chunk_overlap_child

    def merge_hyphenated_words(self, text: str) -> str:
        """
        Merge words in the text that have been split with a hyphen.
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text: str) -> str:
        """
        Replace single newline characters in the text with spaces.
        """
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text: str) -> str:
        """
        Reduce multiple newline characters in the text to a single newline.
        """
        return re.sub(r"\n{2,}", "\n", text)

    def clean_texts(self,
                    texts: List[Tuple[int, str]],
                    cleaning_functions: List[Callable[[str], str]]
                    ) -> List[Tuple[int, str]]:
        """
        Apply the cleaning functions to the text of each page.
        """
        logger.info("Cleaning texts")
        cleaned_texts = []
        for page_num, text in texts:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_texts.append((page_num, text))

        return cleaned_texts

    def texts_to_docs(self,
                      texts: List[Tuple[int, str]],
                      metadata: Dict[str, str]) -> List[docstore.Document]:
        """
        Split the text into chunks and return them as Documents.
        """
        docs: List[docstore.Document] = []
        splitter_language = ut.LANGUAGE_MAP.get(metadata['Language'], 'english')
        splitter = SplitterCreator(self.text_splitter_method,
                                   self.chunk_size,
                                   self.chunk_overlap).get_splitter(splitter_language)
        splitter_child = SplitterCreator(self.text_splitter_method_child,
                                         self.chunk_size_child,
                                         self.chunk_overlap_child).get_splitter(splitter_language)

        prv_page_num = -1
        for page_num, text in texts:
            logger.info(f"Splitting text from page {page_num}")
            # reset chunk number to 0 only when text is from new page
            if page_num != prv_page_num:
                chunk_num = 0
                child_chunk_num = 0
            chunk_texts = splitter.split_text(text)
            # !! chunk_texts can contain duplicates (experienced with ingestion of .txt files)
            # Deduplicate the list chunk_texts
            chunk_texts = list(dict.fromkeys(chunk_texts))
            for chunk_text in chunk_texts:
                # in case of parent retriever, split the parent chunk texts again, into smaller child chunk texts
                # and add parent chunk text as metadata to child chunk text
                if self.retriever_type == "parent":
                    # determine child chunks
                    child_chunk_texts = splitter_child.split_text(chunk_text)
                    # determine child document to store in the vector database
                    for child_chunk_text in child_chunk_texts:
                        # metadata = {"title": , "author": , "indicator_url": , "indicator_closed": , "filename": ,
                        #             "Language": , "last_change_time": }
                        metadata_combined = {
                            "page_number": page_num,
                            "chunk": child_chunk_num,
                            "parent_chunk_num": chunk_num,
                            "parent_chunk": chunk_text,
                            "parent_chunk_id": f"{metadata['filename']}_p{page_num}_c{chunk_num}",
                            "source": f"p{page_num}-{chunk_num}",
                            **metadata,
                        }
                        doc = docstore.Document(
                            page_content=child_chunk_text,
                            # metadata_combined = {"title": , "author": , "indicator_url": , "indicator_closed": ,
                            #                      "filename": , "Language": , "last_change_time": ,"page_number": ,
                            #                      "chunk": , "parent_chunk_num", "parent_chunk": , "parent_chunk_id",
                            #                      "source": }
                            metadata=metadata_combined
                        )
                        docs.append(doc)
                        child_chunk_num += 1
                else:
                    # metadata = {"title": , "author": , "indicator_url": , "indicator_closed": , "filename": ,
                    #             "Language": , "last_change_time": }
                    metadata_combined = {
                        "page_number": page_num,
                        "chunk": chunk_num,
                        "source": f"p{page_num}-{chunk_num}",
                        **metadata,
                    }
                    doc = docstore.Document(
                        page_content=chunk_text,
                        # metadata_combined = {"title": , "author": , "indicator_url": , "indicator_closed": ,
                        #                      "filename": , "Language": , "last_change_time": , "page_number": ,
                        #                      "chunk": , "source": }
                        metadata=metadata_combined
                    )
                    docs.append(doc)
                chunk_num += 1
                prv_page_num = page_num

        return docs

    def clean_texts_to_docs(self, raw_texts, metadata) -> List[docstore.Document]:
        """"
        Combines the functions clean_text and text_to_docs
        """
        cleaning_functions: List = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines
        ]
        cleaned_texts = self.clean_texts(raw_texts, cleaning_functions)
        # for cleaned_text in cleaned_texts:
        #     cleaned_chunks = self.split_text_into_chunks(cleaned_text, metadata)
        docs = self.texts_to_docs(texts=cleaned_texts,
                                  metadata=metadata)
        return docs

    def count_ada_tokens(self, raw_texts: List[Tuple[int, str]]) -> int:
        """
        Counts the number of tokens in the given text for OpenAI's Ada embedding model.

        Parameters:
            text (str): The input text.

        Returns:
            int: The number of tokens.
        """
        total_tokens = 0
        for _, text in raw_texts:
            # Load the tokenizer for the Ada model
            tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
            # Encode the text and count tokens
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
        return total_tokens

    def ingest(self) -> None:
        """
        Ingests all relevant files in the folder
        Checks are done whether vector store needs to be synchronized with folder contents
        """
        # get embeddings
        embeddings = EmbeddingsCreator(self.embeddings_provider,
                                       self.embeddings_model).get_embeddings()

        # create empty list representing added files
        files_added = []

        # get all relevant files in the folder
        relevant_files_in_folder_selected = ut.get_relevant_files_in_folder(self.content_folder,
                                                                            self.document_selection)
        relevant_files_in_folder_all = ut.get_relevant_files_in_folder(self.content_folder)
        # if the vector store already exists, get the set of ingested files from the vector store
        if os.path.exists(self.vecdb_folder):
            # get vector store
            vector_store = VectorStoreCreator().get_vectorstore(embeddings,
                                                                self.collection_name,
                                                                self.vecdb_folder)
            logger.info(f"Vector store already exists for specified settings and folder {self.content_folder}")
            # determine the files that are added or deleted
            collection = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            files_in_store = [metadata['filename'] for metadata in collection['metadatas']]
            files_in_store = list(set(files_in_store))

            # check if files were added, removed or updated
            # files_added depend on the selection that has been made
            # files_updated depend on the selction that has been made
            # files_deleted depend on all files in the folder
            files_added = [file for file in relevant_files_in_folder_selected if file not in files_in_store]
            files_deleted = [file for file in files_in_store if file not in relevant_files_in_folder_all]
            # Check for last changed date
            filename_lastchange_dict = {metadata['filename']: metadata.get('last_change_time', None)
                                        for metadata in collection['metadatas']}
            files_updated = [file for file in relevant_files_in_folder_selected
                             if (file not in files_added) and
                                (filename_lastchange_dict[file] !=
                                 os.stat(os.path.join(self.content_folder, file)).st_mtime)]

            # delete from vector store all chunks associated with deleted or updated files
            to_delete = files_deleted + files_updated
            if len(to_delete) > 0:
                logger.info(f"Files are deleted, so vector store for {self.content_folder} needs to be updated")
                idx_id_to_delete = []
                for idx in range(len(collection['ids'])):
                    idx_id = collection['ids'][idx]
                    idx_metadata = collection['metadatas'][idx]
                    if idx_metadata['filename'] in to_delete:
                        idx_id_to_delete.append(idx_id)
                        if idx_metadata['filename'].endswith(".docx"):
                            os.remove(os.path.join(self.content_folder,
                                                   "conversions",
                                                   idx_metadata['filename'] + ".pdf"))
                vector_store.delete(idx_id_to_delete)
                logger.info("Deleted files from vectorstore")

            # add to vector store all chunks associated with added or updated files
            to_add = files_added + files_updated
        # else it needs to be created first
        else:
            logger.info(f"Vector store to be created for folder {self.content_folder}")
            # get chroma vector store
            vector_store = VectorStoreCreator().get_vectorstore(embeddings,
                                                                self.collection_name,
                                                                self.vecdb_folder)
            # all relevant files in the folder are to be ingested into the vector store
            to_add = list(relevant_files_in_folder_selected)

        # If there are any files to be ingested into the vector store
        if len(to_add) > 0:
            logger.info(f"Files are added, so vector store for {self.content_folder} needs to be updated")
            # create FileParser object
            file_parser = FileParser()

            for file in to_add:
                file_path = os.path.join(self.content_folder, file)
                # extract raw text pages and metadata according to file type
                logger.info(f"Parsing file {file}")
                raw_texts, metadata = file_parser.parse_file(file_path)
                documents = self.clean_texts_to_docs(raw_texts=raw_texts,
                                                     metadata=metadata)
                # count tokens
                tokens_document = self.count_ada_tokens(raw_texts)
                logger.info(f"Extracted {len(documents)} chunks (Tokens: {tokens_document}) from {file}")
                if tokens_document > 40_000:
                    logger.info("Pause for 10 seconds to avoid hitting rate limit")
                    time.sleep(10)
                vector_store.add_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.vecdb_folder,
                )
            logger.info("Added files to vectorstore")
