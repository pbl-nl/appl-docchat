import os
from loguru import logger
# LLM modules
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# local imports
import settings


class LLMCreator():
    """
    LLM class to import into other modules
    """
    def __init__(self, llm_type=None, llm_model_type=None) -> None:
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type

    def get_llm(self):
        """
        returns, based on settings, the llm object
        """
        # if llm_type is "chatopenai"
        if self.llm_type == "chatopenai":
            # default llm_model_type value is "gpt-3.5-turbo"
            self.llm_model_type = "gpt-3.5-turbo"
            if self.llm_model_type == "gpt35_16":
                self.llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                self.llm_model_type = "gpt-4"
            llm = ChatOpenAI(
                client=None,
                model=self.llm_model_type,
                temperature=0,
            )
        # else, if llm_type is "huggingface"
        elif self.llm_type == "huggingface":
            if self.llm_model_type == "google/flan-t5-base":
                max_length = 512
            llm = HuggingFaceHub(repo_id=self.llm_model_type,
                                 model_kwargs={"temperature": 0,
                                               "max_length": max_length}
                                 )
        # else, if llm_type is "ollama"
        elif self.llm_type == "ollama":
            logger.info("Use Local LLM")
            logger.info("Retrieving " + self.llm_model_type)
            llm = Ollama(
                model=self.llm_model_type,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
            logger.info("Retrieved " + self.llm_model_type)
        # else, if llm_type is "azurechatopenai"
        elif self.llm_type == "azurechatopenai":
            logger.info("Use Azure OpenAI LLM")
            logger.info("Retrieving " + self.llm_model_type)
            llm = AzureChatOpenAI(model=self.llm_model_type,
                                  azure_deployment=os.environ["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
                                  api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                  temperature=0)
            logger.info("Retrieved " + self.llm_model_type)

        return llm
