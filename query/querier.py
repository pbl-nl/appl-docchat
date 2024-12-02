from typing import Dict, Any, List
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pydantic import Field
from dotenv import load_dotenv
# local imports
import settings
from ingest.embeddings_creator import EmbeddingsCreator
from ingest.vectorstore_creator import VectorStoreCreator
from query.llm_creator import LLMCreator
from query.retriever_creator import RetrieverCreator
import prompts.prompt_templates as pr
import utils as ut


class Querier:
    """
    When parameters are read from settings.py, object is initiated without parameter settings
    When parameters are read from GUI, object is initiated with parameter settings listed
    """
    def __init__(self, llm_provider=None, llm_model=None, embeddings_provider=None, embeddings_model=None,
                 vecdb_type=None, chain_name=None, chain_type=None, chain_verbosity=None, search_type=None,
                 score_threshold=None, chunk_k=None):
        load_dotenv()
        self.llm_provider = settings.LLM_PROVIDER if llm_provider is None else llm_provider
        self.llm_model = settings.LLM_MODEL if llm_model is None else llm_model
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chain_name = settings.CHAIN_NAME if chain_name is None else chain_name
        self.chain_type = settings.CHAIN_TYPE if chain_type is None else chain_type
        self.chain_verbosity = settings.CHAIN_VERBOSITY if chain_verbosity is None else chain_verbosity
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.score_threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.chat_history = []
        self.vector_store = None
        self.chain = None

        # define llm
        self.llm = LLMCreator(self.llm_provider,
                              self.llm_model).get_llm()

        # define embeddings
        self.embeddings = EmbeddingsCreator(self.embeddings_provider,
                                            self.embeddings_model).get_embeddings()

    def make_chain(self,
                   content_folder: str,
                   vecdb_folder: str,
                   search_filter: Dict = None) -> None:
        """
        Creates the chain that is used for question answering

        Parameters
        ----------
        content_folder : str
            the content folder
        vecdb_folder : str
            the folder of the vector databse that is associated with the content folder
        search_filter : Dict, optional
            _description_, by default None
        """
        # get vector store
        self.vector_store = VectorStoreCreator(self.vecdb_type).get_vectorstore(embeddings=self.embeddings,
                                                                                content_folder=content_folder,
                                                                                vecdb_folder=vecdb_folder)
        logger.info(f"Loaded vector store from folder {vecdb_folder}")

        # get retriever with search_filter
        self.retriever = RetrieverCreator(vectorstore=self.vector_store).get_retriever(search_filter=search_filter)

        # get appropriate RAG prompt for querying
        if settings.RETRIEVER_PROMPT_TEMPLATE == "openai_rag":
            current_template = pr.OPENAI_RAG_TEMPLATE
        elif settings.RETRIEVER_PROMPT_TEMPLATE == "openai_rag_concise":
            current_template = pr.OPENAI_RAG_CONCISE_TEMPLATE
        elif settings.RETRIEVER_PROMPT_TEMPLATE == "openai_rag_language":
            current_template = pr.OPENAI_RAG_LANGUAGE_TEMPLATE
        elif settings.RETRIEVER_PROMPT_TEMPLATE == "yesno":
            current_template = pr.YES_NO_TEMPLATE
        prompt = PromptTemplate.from_template(template=current_template)

        # get chain
        if self.chain_name == "conversationalretrievalchain":
            # Create custom class to have more control over retrieval process
            class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
                context: List[Any] = Field(default_factory=list)

                def _get_docs(self, *args, **kwargs):
                    # Replace this with your custom logic for retrieving documents
                    return self.context

            self.chain = CustomConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                chain_type=self.chain_type,
                verbose=self.chain_verbosity,
                combine_docs_chain_kwargs={'prompt': prompt},
                return_source_documents=True
            )
        logger.info("Executed Querier.make_chain")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Finds most similar docs to prompt in the vectorstore and determines the response
        If the closest doc found is not similar enough to the prompt, any answer from the LLM is overruled by a message

        Parameters
        ----------
        question : str
            the question that was asked by the user

        Returns
        -------
        Dict[str, Any]
            the response from the chain, containing the answer to the question and the sources used
        """
        logger.info(f"current question: {question}")
        logger.info(f"current chat history: {self.chat_history}")

        # Detects languages, translates question, retrieves context
        # languages = set of tuples of language code and language
        languages = ut.retrieve_languages_from_vector_store(vector_store=self.vector_store)
        chunks = []
        for language in languages:
            # determine the files in that language
            all_documents = self.vector_store.get()
            relevant_files = set()
            for metadata in all_documents['metadatas']:
                if metadata["Language"] == language[0]:
                    relevant_files.add(metadata["filename"])
            # define filter based on filenames
            docfilter = {"filename": {"$in": list(relevant_files)}}

            translation_instruction = \
                f"""Translate the following text to {language[1]} and only return translation: \n {question}"""
            # ask llm to translate language of prompt to current language
            # if language of prompt is not equal to current language
            prompt_language = ut.detect_language(text=question)
            if language[0] != prompt_language:
                translated_question = self.llm.invoke(input=translation_instruction)
                chunks.extend(self.retriever.invoke(input=translated_question.content, search_filter=docfilter))
            else:
                chunks.extend(self.retriever.invoke(input=question, search_filter=docfilter))

        self.chain.context = sorted(chunks, key=lambda doc: doc.metadata['relevance_score'],
                                    reverse=True)[:min(self.chunk_k, len(chunks))]

        response = self.chain.invoke({"question": question, "chat_history": self.chat_history})
        # if no chunk qualifies, overrule any answer generated by the LLM
        if len(response["source_documents"]) == 0:
            language = ut.detect_language(text=question)
            response["answer"] = ut.get_no_response_answer(language)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))

        return response

    def clear_history(self) -> None:
        """
        Clears the chat history
        Used by "Clear Conversation" button in streamlit_app.py
        """
        self.chat_history = []

    def get_meta_data_by_file_name(self, filename: str) -> Dict[str, str]:
        """
        Returns the meta data of a specific file
        Need to run make_chain first

        Parameters
        ----------
        filename : str
            the filename used to refer to get all chunk metadata

        Returns
        -------
        Dict[str: str]
            chunks metadata like filename, pagenumber, etc
        """
        # sources keys: ['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data']
        sources = self.vector_store.get()
        metadata = [metadata for metadata in sources['metadatas'] if metadata['filename'] == filename]

        # return only the first chunk, as filename metadata is the same for all chunks
        return metadata[0]
