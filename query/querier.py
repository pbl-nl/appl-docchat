from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from loguru import logger
# local imports
import settings


class Querier:
    def __init__(self):
        load_dotenv()
        self.chat_history = []

    def make_chain(self, input_folder, vectordb_folder):
        self.input_folder = input_folder
        self.vectordb_folder = vectordb_folder

        if settings.LLM_TYPE == "chatopenai":
            if settings.LLM_MODEL_TYPE == "gpt35":
                llm_model_type = "gpt-3.5-turbo"
            elif settings.LLM_MODEL_TYPE == "gpt35_16":
                llm_model_type = "gpt-3.5-turbo-16k"
            elif settings.LLM_MODEL_TYPE == "gpt4":
                llm_model_type = "gpt-4"
            llm = ChatOpenAI(
                client=None,
                model=llm_model_type,
                temperature=0,
            )

        if settings.EMBEDDINGS_PROVIDER == "openai":
            embeddings = OpenAIEmbeddings(model=settings.EMBEDDINGS_MODEL, client=None)
            logger.info("Loaded openai embeddings")

        if settings.VECDB_TYPE == "chromadb":
            vector_store = Chroma(
                collection_name=self.input_folder,
                embedding_function=embeddings,
                persist_directory=self.vectordb_folder,
            )
            retriever = vector_store.as_retriever(search_type=settings.SEARCH_TYPE, search_kwargs={"k": settings.CHUNK_K})
            logger.info(f"Loaded chromadb from folder {self.vectordb_folder}")


        if settings.CHAIN == "conversationalretrievalchain":
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type=settings.CHAIN_TYPE,
                verbose=settings.CHAIN_VERBOSITY,
                return_source_documents=True
            )

        logger.info("Executed Querier.make_chain(self, input_folder, vectordb_folder)")
        self.chain = chain

    def ask_question(self, question: str):
        logger.info(f"current chat history: {self.chat_history}")
        response = self.chain({"question": question, "chat_history": self.chat_history})
        logger.info(f"question: {question}")

        answer = response["answer"]
        logger.info(f"answer: {answer}")
        source = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return response
    
    def clear_history(self):
        # used by "Clear Conversation" button
        self.chat_history = []

    
 
