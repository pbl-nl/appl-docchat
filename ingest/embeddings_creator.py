from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# local imports
import settings


class EmbeddingsCreator():
    """
    LLM class to import into other modules
    """
    def __init__(self, embeddings_provider=None, embeddings_model=None, local_api_url=None,
                 azureopenai_api_version=None) -> None:
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.local_api_url = settings.API_URL if local_api_url is None else local_api_url
        self.azureopenai_api_version = settings.AZUREOPENAI_API_VERSION \
            if azureopenai_api_version is None and settings.AZUREOPENAI_API_VERSION is not None \
            else azureopenai_api_version

    def get_embeddings(self):
        """
        returns, based on settings, the embeddings object
        """
        # determine embeddings model
        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, client=None)
            logger.info("Loaded openai embeddings")
        elif self.embeddings_provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        elif self.embeddings_provider == "local_embeddings":
            if self.local_api_url is not None:
                embeddings = OllamaEmbeddings(
                    base_url=self.local_api_url,
                    model=self.embeddings_model)
            else:
                embeddings = OllamaEmbeddings(
                    model=self.embeddings_model)
            logger.info("Loaded local embeddings: " + self.embeddings_model)
        elif self.embeddings_provider == "azureopenai":
            logger.info("Retrieve " + self.embeddings_model)
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.embeddings_model,
                openai_api_version=self.azureopenai_api_version,
                azure_endpoint=self.local_api_url,
                )
            logger.info("Loaded Azure OpenAI embeddings")

        return embeddings
