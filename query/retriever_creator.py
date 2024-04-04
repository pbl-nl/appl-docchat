# imports
from loguru import logger
# LLM modules
# local imports
import settings_template as settings


class RetrieverCreator():
    """
    Retriever class to import into other modules
    """
    def __init__(self, vectorstore, search_filter, retriever_type=None, chunk_k=None, search_type=None,
                 score_threshold=None) -> None:
        # self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.score_threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.vectorstore = vectorstore
        self.search_filter = search_filter

    def get_retriever(self):
        """
        returns, based on settings, the retriever object
        """
        # if retriever_type is "vectorstore"
        if self.retriever_type == "vectorstore":
            # get retriever with some search arguments
            # maximum number of chunks to retrieve
            search_kwargs = {"k": self.chunk_k}
            # filter, if set
            if self.search_filter is not None:
                logger.info(f"querying vector store with filter {self.search_filter}")
                search_kwargs["filter"] = self.search_filter
            if self.search_type == "similarity_score_threshold":
                search_kwargs["score_threshold"] = self.score_threshold

            retriever = self.vectorstore.as_retriever(search_type=self.search_type, search_kwargs=search_kwargs)
        # # else, if llm_type is "parent"
        # elif self.retriever_type == "parent":

        logger.info("Set retriever to" + retriever)

        return retriever
