# imports
from loguru import logger
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
# local imports
import settings


class RetrieverCreator():
    """
    Retriever class to import into other modules
    """
    # def __init__(self, vectorstore, search_kwargs, retriever_type=None, chunk_k=None, search_type=None,
    #              score_threshold=None, multiquery=None) -> None:
    def __init__(self, vectorstore, retriever_type=None, chunk_k=None, search_type=None,
                 score_threshold=None, multiquery=None) -> None:
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.score_threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.multiquery = settings.MULTIQUERY if multiquery is None else multiquery
        self.vectorstore = vectorstore
        # self.search_kwargs = search_kwargs

    def get_retriever(self, search_filter=None):
        """
        returns, based on the RETRIEVER_TYPE settings, the retriever object
        """
        if self.retriever_type == "vectorstore":
            # maximum number of chunks to retrieve
            search_kwargs = {"k": self.chunk_k}
            # filter, if set
            if search_filter is not None:
                logger.info(f"querying vector store with filter {search_filter}")
                search_kwargs["filter"] = search_filter
            if self.search_type == "similarity_score_threshold":
                search_kwargs["score_threshold"] = self.score_threshold

            # retriever = self.vectorstore.as_retriever(search_type=self.search_type,
            #                                           search_kwargs=self.search_kwargs)
            retriever = self.vectorstore.as_retriever(search_type=self.search_type,
                                                      search_kwargs=search_kwargs)
        # elif self.retriever_type == "parent":

        logger.info(f"Set retriever to {self.retriever_type}")

        if self.multiquery:
            # the llm to create multiple questions from the user question
            llm = ChatOpenAI(temperature=0)
            # multiqueryretriever creates 3 queries iso 1
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm, include_original=True)
            logger.info("Using multiple queries")

        return retriever
