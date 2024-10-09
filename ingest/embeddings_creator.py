import os
from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# local imports
import settings


class EmbeddingsCreator():
    """
    EmbeddingsCreator class to import into other modules
    """
    def __init__(self, embeddings_provider: str = None, embeddings_model: str = None) -> None:
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model

    def get_embeddings(self):
        """
        returns, based on settings, the embeddings object
        """
        # determine embeddings model
        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model)
            logger.info(f"Loaded openai embeddings model {self.embeddings_model}")
        elif self.embeddings_provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
            logger.info(f"Loaded huggingface embeddings model {self.embeddings_model}")
        elif self.embeddings_provider == "ollama":
            embeddings = OllamaEmbeddings(model=self.embeddings_model)
            logger.info(f"Loaded local embeddings model {self.embeddings_model}")
        elif self.embeddings_provider == "azureopenai":
            embeddings = AzureOpenAIEmbeddings(model=self.embeddings_model,
                                               azure_deployment=os.environ["AZURE_OPENAI_EMB_DEPLOYMENT_NAME"],
                                               api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                               client=None)
            logger.info(f"Loaded Azure OpenAI embeddings model {self.embeddings_model}")

        return embeddings
