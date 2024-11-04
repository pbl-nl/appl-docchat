import os
from loguru import logger
# LLM modules
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# local imports
import settings


class LLMCreator():
    """
    LLM class to import into other modules
    """
    def __init__(self, llm_provider=None, llm_model=None) -> None:
        self.llm_provider = settings.LLM_PROVIDER if llm_provider is None else llm_provider
        self.llm_model = settings.LLM_MODEL if llm_model is None else llm_model

    def get_llm(self):
        """
        returns, based on settings, the llm object
        """
        llm = None
        if self.llm_provider == "openai":
            logger.info("Use OpenAI LLM")
            llm = ChatOpenAI(
                client=None,
                model=self.llm_model,
                temperature=0,
            )
        elif self.llm_provider == "huggingface":
            logger.info("Use HuggingFace LLM")
            if self.llm_model == "google/flan-t5-base":
                max_length = 512
            llm = HuggingFaceHub(repo_id=self.llm_model,
                                 model_kwargs={"temperature": 0,
                                               "max_length": max_length}
                                 )
        elif self.llm_provider == "ollama":
            logger.info("Use Ollama local LLM")
            llm = Ollama(
                model=self.llm_model,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
        elif self.llm_provider == "azureopenai":
            logger.info("Use Azure OpenAI LLM")
            llm = AzureChatOpenAI(model=self.llm_model,
                                  azure_deployment=os.environ["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
                                  api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                  temperature=0)
        logger.info(f"Retrieved model: {self.llm_model}")

        return llm
