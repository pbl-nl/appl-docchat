from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from loguru import logger

# from settings import COLLECTION_NAME, PERSIST_DIRECTORY


class Querier:
    def __init__(self, input_folder: str, vectordb_folder:str, embeddings_type: str, vectordb_type: str, chunk_size: int, chunk_overlap: int):
        load_dotenv()
        self.input_folder = input_folder
        self.vectordb_folder = vectordb_folder
        self.vectordb_type = vectordb_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_type = embeddings_type
        self.chain = self.make_chain()
        self.chat_history = []

    def make_chain(self):
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

        # WERKT
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            return_source_documents=True,
        )

        # retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm,
        #     retriever=retriever,
        #     memory=memory,
        #     return_source_documents=True,
        # )
        return chain

    def ask_question(self, question: str):
        response = self.chain({"question": question, "chat_history": self.chat_history})
        logger.info(f"current chat history: {self.chat_history}")

        answer = response["answer"]
        source = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return answer, source
