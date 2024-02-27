# imports
from loguru import logger
# LLM modules
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# local imports
import settings_template as settings


class LLM():
    '''LLM class to import into other modules'''
    def __init__(self, llm_type=None, llm_model_type=None, local_api_url=None, azureopenai_api_version=None) -> None:
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.local_api_url = settings.API_URL if local_api_url is None else local_api_url
        self.azureopenai_api_version = settings.AZUREOPENAI_API_VERSION \
            if azureopenai_api_version is None and settings.AZUREOPENAI_API_VERSION is not None \
            else azureopenai_api_version

        # if llm_type is "chatopenai"
        if self.llm_type == "chatopenai":
            # default llm_model_type value is "gpt-3.5-turbo"
            self.llm_model_type = "gpt-3.5-turbo"
            if self.llm_model_type == "gpt35_16":
                self.llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                self.llm_model_type = "gpt-4"
            self.llm = ChatOpenAI(
                client=None,
                model=self.llm_model_type,
                temperature=0,
            )
        # else, if llm_type is "huggingface"
        elif self.llm_type == "huggingface":
            # default value is llama-2, with maximum output length 512
            self.llm_model_type = "meta-llama/Llama-2-7b-chat-hf"
            max_length = 512
            if self.llm_model_type == 'GoogleFlan':
                self.llm_model_type = 'google/flan-t5-base'
                max_length = 512
            self.llm = HuggingFaceHub(repo_id=self.llm_model_type,
                                      model_kwargs={"temperature": 0.1,
                                                    "max_length": max_length}
                                      )
        # else, if llm_type is "local_llm"
        elif self.llm_type == "local_llm":
            logger.info("Use Local LLM")
            logger.info("Retrieving " + self.llm_model_type)
            if self.local_api_url is not None:  # If API URL is defined, use it
                logger.info("Using local api url " + self.local_api_url)
                self.llm = Ollama(
                    model=self.llm_model_type,
                    base_url=self.local_api_url,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            else:
                self.llm = Ollama(
                    model=self.llm_model_type,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            logger.info("Retrieved " + self.llm_model_type)

        # else, if llm_type is "azureopenai"
        elif self.llm_type == "azureopenai":
            logger.info("Use Azure OpenAI LLM")
            logger.info("Retrieving " + self.llm_model_type)
            self.llm = AzureChatOpenAI(
                azure_deployment=self.llm_model_type,
                azure_endpoint=self.local_api_url,
                api_version=self.azureopenai_api_version,
            )
            logger.info("Retrieved " + self.llm_model_type)

    def get_llm(self):
        return self.llm
