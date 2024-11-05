"""
Ingester class
Creates Ingester object
Also parses files, chunks the files and stores the chunks in vector store
When instantiating without parameters, attributes get values from settings.py
"""
import os
import re
from typing import Callable, Dict, List, Tuple, Any
from loguru import logger
import langchain.docstore.document as docstore
from dotenv import load_dotenv
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
                 embeddings_provider: str = None, embeddings_model: str = None,
                 retriever_type: str = None, vecdb_type: str = None,
                 text_splitter_method: str = None, text_splitter_method_child: str = None,
                 chunk_size: int = None, chunk_size_child: int = None,
                 chunk_overlap: int = None, chunk_overlap_child: int = None) -> None:
        load_dotenv()
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vecdb_folder = vecdb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
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
                      embeddings: Any,
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
            chunk_texts = splitter.split_text(text)
            for chunk_text in chunk_texts:
                # in case of parent retriever, split the parent chunk texts again, into smaller child chunk texts
                # and add parent chunk text as metadata to child chunk text
                if self.retriever_type == "parent":
                    # determine parent chunk embedding. It needs to be stored as a string in the vector database
                    parent_chunk_embedding = ','.join(str(x) for x in embeddings.embed_documents([chunk_text])[0])
                    # determine child chunks
                    child_chunk_texts = splitter_child.split_text(chunk_text)
                    # determine child document to store in the vector database
                    for child_chunk_text in child_chunk_texts:
                        # metadata = {"title": , "author": , "indicator_url": , "indicator_closed": , "filename": ,
                        #             "Language": }
                        metadata_combined = {
                            "page_number": page_num,
                            "parent_chunk_num": chunk_num,
                            "parent_chunk": chunk_text,
                            "parent_chunk_id": f"{metadata['filename']}_p{page_num}_c{chunk_num}",
                            "parent_chunk_embedding": parent_chunk_embedding,
                            "source": f"p{page_num}-{chunk_num}",
                            **metadata,
                        }
                        doc = docstore.Document(
                            page_content=child_chunk_text,
                            # metadata_combined = {"title": , "author": , "indicator_url": , "indicator_closed": ,
                            #                      "filename": , "Language": , "page_number": , "chunk": ,
                            #                      "parent_chunk": , "parent_chunk_id", "parent_chunk_embedding: ,
                            #                      "source": }
                            metadata=metadata_combined
                        )
                        docs.append(doc)
                else:
                    # metadata = {"title": , "author": , "indicator_url": , "indicator_closed": , "filename": ,
                    #             "Language": }
                    metadata_combined = {
                        "page_number": page_num,
                        "chunk": chunk_num,
                        "source": f"p{page_num}-{chunk_num}",
                        **metadata,
                    }
                    doc = docstore.Document(
                        page_content=chunk_text,
                        # metadata_combined = {"title": , "author": , "indicator_url": , "indicator_closed": ,
                        # "filename": , "Language": , "page_number": , "chunk": , "source": }
                        metadata=metadata_combined
                    )
                    docs.append(doc)
                chunk_num += 1
                prv_page_num = page_num

        return docs

    def clean_texts_to_docs(self, raw_texts, embeddings, metadata) -> List[docstore.Document]:
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
        docs = self.texts_to_docs(cleaned_texts, embeddings, metadata)

        return docs

    def ingest(self) -> None:
        """
        Ingests all relevant files in the folder
        Checks are done whether vector store needs to be synchronized with folder contents
        """
        # get embeddings
        embeddings = EmbeddingsCreator(self.embeddings_provider,
                                       self.embeddings_model).get_embeddings()

        # create empty list representing added files
        new_files = []

        # get all relevant files in the folder
        relevant_files_in_folder = ut.get_relevant_files_in_folder(self.content_folder)
        # if the vector store already exists, get the set of ingested files from the vector store
        if os.path.exists(self.vecdb_folder):
            # get chroma vector store
            vector_store = VectorStoreCreator(self.vecdb_type).get_vectorstore(embeddings,
                                                                               self.collection_name,
                                                                               self.vecdb_folder)
            logger.info(f"Vector store already exists for specified settings and folder {self.content_folder}")
            # determine the files that are added or deleted
            collection = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            files_in_store = [metadata['filename'] for metadata in collection['metadatas']]
            files_in_store = list(set(files_in_store))

            # get all converted files that are relevant (.docx files are converted to .pdf and stored under conversions)
            converted_folder_path = os.path.join(self.content_folder,'conversions')
            to_be_updated = []
            if os.path.exists(converted_folder_path):
                converted_files = [f.split('.pdf')[0] #removing .pdf to be left with .docx etc.
                                for f in ut.get_relevant_files_in_folder(converted_folder_path) 
                                if f.endswith('.pdf')] # ensure that user did not add non pdf file to conversions
                
                # Check for last changed date of non-pdf files
                filename_lastchange_dict = { metadata['filename']: metadata.get('last_change_time', None)
                                                for metadata in collection['metadatas'] }
                to_be_updated = [ file for file in converted_files 
                    if (os.path.exists(os.path.join(self.content_folder, file))) and # ensure that original file is not removed
                    (filename_lastchange_dict[file+'.pdf'] != os.stat(os.path.join(self.content_folder, file)).st_mtime) ]

                # if .docx file is not converted to pdf before, it should be kept as .docx to be converted and
                # ingested. Otherwise, keep the file as .docx.pdf
                relevant_files_in_folder = [f if f not in converted_files else f'{f}.pdf' #.docx.pdf
                                            for f in relevant_files_in_folder]
                
            # check if files were added or removed
            new_files = [file for file in relevant_files_in_folder if file not in files_in_store]
            files_deleted = [file for file in files_in_store if file not in relevant_files_in_folder]
            
            # Synchronize the vector store and conversions (pdf copy of .docx etc.)
            files_to_be_removed = [file for file in files_deleted 
                                   if os.path.splitext(os.path.splitext(file)[0])[1] in [".docx", ".txt", ".html", ".md"]]
            for file in files_to_be_removed:
                converted_file_path = os.path.join(converted_folder_path, file)
                if os.path.exists(converted_file_path): #make sure user did not remove manually
                    os.remove(converted_file_path)

            new_files.extend(to_be_updated) 
            files_deleted.extend(to_be_updated) # before adding the files old information needs to be removed from vector store

            # delete all chunks from the vector store that belong to files removed from the folder
            if len(files_deleted) > 0:
                logger.info(f"Files are deleted, so vector store for {self.content_folder} needs to be updated")
                idx_id_to_delete = []
                for idx in range(len(collection['ids'])):
                    idx_id = collection['ids'][idx]
                    idx_metadata = collection['metadatas'][idx]
                    if idx_metadata['filename'] in files_deleted:
                        idx_id_to_delete.append(idx_id)
                vector_store.delete(idx_id_to_delete)
                logger.info("Deleted files from vectorstore")
        # else it needs to be created first
        else:
            logger.info(f"Vector store to be created for folder {self.content_folder}")
            # get chroma vector store
            vector_store = VectorStoreCreator(self.vecdb_type).get_vectorstore(embeddings,
                                                                               self.collection_name,
                                                                               self.vecdb_folder)
            # all relevant files in the folder are to be ingested into the vector store
            new_files = list(relevant_files_in_folder)

        # If there are any files to be ingested into the vector store
        if len(new_files) > 0:
            logger.info(f"Files are added, so vector store for {self.content_folder} needs to be updated")
            # create FileParser object
            file_parser = FileParser()

            for file in new_files:
                file_path = os.path.join(self.content_folder, file)
                # extract raw text pages and metadata according to file type
                raw_texts, metadata = file_parser.parse_file(file_path)
                documents = self.clean_texts_to_docs(raw_texts, embeddings, metadata)
                logger.info(f"Extracted {len(documents)} chunks from {file}")
                vector_store.add_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.vecdb_folder,
                )
            logger.info("Added files to vectorstore")
