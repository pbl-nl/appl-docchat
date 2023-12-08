import sys
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.utilities import SerpAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.utilities import TextRequestsWrapper
from dotenv import load_dotenv
from loguru import logger
import requests
# local imports
import settings


def get_geocode_address(query):
    #make a account in geonames website and change the username into your account
    username = "demo"
    url = f"http://api.geonames.org/geoCodeAddressJSON?q={query}&username={username}"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        return response.json()
    else:
        return None
class CustomTool(BaseTool):
    name = "CustomTool"
    description = "useful for when you need to answer questions about the coordinates of an address"
    def _run(self, query: str) -> str:
        """Use the tool."""
        return get_geocode_address(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("API does not support async")

tools = [
    Tool(
        name="GeoCodeAddress",
        func=CustomTool.run,
        description="useful for when you need to answer questions about the coordinates of an address"
    )
    ]

tools = [CustomTool()]

class AgentChain():
    def __init__(self, llm_type=None, llm_model_type=None):
        load_dotenv()
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.chat_history = []

    def get_input(self):
        """ This function will be called when the agent does not know the answer and will ask for human inputs"""
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)

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

        self.geo_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        # self.wiki_agent = initialize_agent(tools,
        #                                    llm,
        #                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #                                    verbose=True,
        #                                    return_intermediate_steps=True,
        #                                    )

    def get_nearby_location(self, lat, lng, username):
        url = f"https://api.geonames.org/extendedFindNearby?lat={lat}&lng={lng}&username={username}"
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def ask_question(self, question: str):
        logger.info(f"current chat history: {self.chat_history}")
        # response = self.agent_executor.invoke({"input": question, "chat_history": self.chat_history})
        # response = self.wiki_agent(question)
        response = self.geo_agent(question)
        final_answer = response["output"]
        # logger.info(f"intermediate steps: {response['intermediate_steps']}")
        logger.info(f"question: {question}")
        logger.info(f"answer: {final_answer}")
        # self.chat_history.append(HumanMessage(content=question))
        # self.chat_history.append(AIMessage(content=final_answer))
        return final_answer

    def clear_history(self):
        # used by "Clear Conversation" button
        self.chat_history = []

def exit_program():
    print("Exiting the program...")
    sys.exit(0)


def main():
    agent_chain = AgentChain()
    agent_chain.make_agent()
    while True:
        question = input("Question: ")
        if question not in ["exit", "quit", "q"]:
            # log the question
            logger.info(f"\nQuestion: {question}")
            # use agent to generate answer
            response = agent_chain.ask_question(question)
            logger.info(f"\nAnswer: {response}")
        else:
            exit_program()
    return

if __name__ == "__main__":
    main()
