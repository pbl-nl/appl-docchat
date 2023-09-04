from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from loguru import logger


class Querier:
    def __init__(self, embeddings_type: str, vectordb_type: str, chunk_size: int, chunk_overlap: int):
        load_dotenv()
        self.vectordb_type = vectordb_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_type = embeddings_type
        self.chat_history = []

    def make_chain(self, input_folder, vectordb_folder):
        self.input_folder = input_folder
        self.vectordb_folder = vectordb_folder

        llm = ChatOpenAI(
            client=None,
            model="gpt-3.5-turbo",
            temperature=0,
        )

        if self.embeddings_type == "openai":
            embeddings = OpenAIEmbeddings(client=None)
            logger.info("Loaded openai embeddings")

        if self.vectordb_type == "chromadb":
            vector_store = Chroma(
                collection_name=self.input_folder,
                embedding_function=embeddings,
                persist_directory=self.vectordb_folder,
            )
            logger.info(f"Loaded chromadb from folder {self.vectordb_folder}")

        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            return_source_documents=True,
        )

        logger.info("Executed make_chain(self, input_folder, vectordb_folder)")
        # return chain
        self.chain = chain

    def ask_question(self, question: str):
        response = self.chain({"question": question, "chat_history": self.chat_history})
        logger.info(f"current chat history: {self.chat_history}")

        answer = response["answer"]
        source = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return answer, source
    
    def clear_history(self):
        # used by "Clear Conversation" button
        self.chat_history = []

    
 
