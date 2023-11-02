import sys
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv
from loguru import logger
# local imports
import settings


class AgentChain():
    def __init__(self, llm_type=None, llm_model_type=None, embeddings_provider=None, embeddings_model=None,
                 vecdb_type=None, chain_name=None, chain_type=None, chain_verbosity=None, search_type=None,
                 chunk_k=None):
        load_dotenv()
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chain_name = settings.CHAIN_NAME if chain_name is None else chain_name
        self.chain_type = settings.CHAIN_TYPE if chain_type is None else chain_type
        self.chain_verbosity = settings.CHAIN_VERBOSITY if chain_verbosity is None else chain_verbosity
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.chat_history = []


    def make_agent(self):
        """Create a langchain agent with selected llm and tools"""
        if self.llm_type == "chatopenai":
            if self.llm_model_type == "gpt35":
                llm_model_type = "gpt-3.5-turbo"
            elif self.llm_model_type == "gpt35_16":
                llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                llm_model_type = "gpt-4"
            llm = ChatOpenAI(
                client=None,
                model=llm_model_type,
                temperature=0,
            )
        #Todo use the custom tool
        tools = load_tools(['wikipedia'], llm=llm)

        #Todo use multiple tools?!
        self.wiki_agent = initialize_agent(tools,
                                           llm,
                                           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                           verbose=True)


    def ask_question_agent(self, question: str):
        logger.info(f"current chat history: {self.chat_history}")
        # response = self.agent_executor.invoke({"input": question, "chat_history": self.chat_history})
        # response = self.agent_executor.invoke({"input": question})
        response = self.wiki_agent.run(question)
        # response = self.chain({"question": question, "chat_history": self.chat_history})
        logger.info(f"question: {question}")
        logger.info(f"answer: {response}")
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        return response

    def clear_history(self):
        # used by "Clear Conversation" button
        self.chat_history = []

def exit_program():
    print("Exiting the program...")
    sys.exit(0)


def main():
    wiki_agent = AgentChain()
    wiki_agent.make_agent()
    while True:
        # Get question from user
        question = input("Question: ")
        if question not in ["exit", "quit", "q"]:
            # log the question
            logger.info(f"\nQuestion: {question}")
            # use agent to generate answer
            response = wiki_agent.ask_question_agent(question)
            logger.info(f"\nAnswer: {response}")
        else:
            exit_program()


if __name__ == "__main__":
    main()
