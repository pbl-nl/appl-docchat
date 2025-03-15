"""
Retriever class to import into other modules
Implementations: vectorstore retriever, ensemble retriever of vectorstore and BM25, custom parent document retriever,
compression retriever combining flashrank reranker with any of the above mentioned retrievers,
multiqueryretriever using multiple versions of the same query with one of the abovementioned retrievers
"""
# imports
from typing import List
from loguru import logger
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_openai import ChatOpenAI
from flashrank import Ranker
# local imports
import settings
from query.retrieve_parent_chunks import ParentDocumentRetriever


class RetrieverCreator():
    """
    Retriever class to import into other modules
    """
    def __init__(self, vectorstore: VectorStore, retriever_type: str = None, retriever_weights: List[float] = None,
                 chunk_k: int = None, chunk_k_child: int = None, search_type: str = None,
                 score_threshold: float = None, rerank: bool = None, rerank_provider: str = None,
                 rerank_model: str = None, chunk_k_for_rerank: int = None, multiquery: bool = None) -> None:
        self.vectorstore = vectorstore
        self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.retriever_weights = settings.RETRIEVER_WEIGHTS if retriever_weights is None else retriever_weights
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.chunk_k_child = settings.CHUNK_K_CHILD if chunk_k_child is None else chunk_k_child
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.score_threshold = settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.rerank = settings.RERANK if rerank is None else rerank
        self.rerank_provider = settings.RERANK_PROVIDER if rerank_provider is None else rerank_provider
        self.rerank_model = settings.RERANK_MODEL if rerank_model is None else rerank_model
        self.chunk_k_for_rerank = settings.CHUNK_K_FOR_RERANK if chunk_k_for_rerank is None else chunk_k_for_rerank
        self.multiquery = settings.MULTIQUERY if multiquery is None else multiquery

    def get_vectorstore_retriever(self, search_filter=None):
        """
        returns plain vectorstore retriever
        """
        # maximum number of chunks to retrieve
        search_kwargs = {"k": self.chunk_k if not self.rerank else self.chunk_k_for_rerank}
        # filter, if set
        if search_filter is not None:
            # logger.info(f"querying vector store with filter {search_filter}")
            search_kwargs["filter"] = search_filter
        if self.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.score_threshold
        retriever = self.vectorstore.as_retriever(search_type=self.search_type,
                                                  search_kwargs=search_kwargs)

        return retriever

    def get_ensemble_retriever(self, search_filter=None):
        """
        returns hybrid search retriever with BM25 algorithm for sparse vector search
        """
        # For BM25 retriever, a search filter on filename cannot directly be used
        # So first create a temporary collection with chunks of just the one file in the searchfilter
        if search_filter is not None:
            # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            collection = self.vectorstore.get()
            filtered_collection = {}
            filtered_collection["documents"] = []
            filtered_collection["metadatas"] = []
            for i, chunk_metadata in enumerate(collection["metadatas"]):
                if chunk_metadata["filename"] == search_filter['filename']:
                    filtered_collection["metadatas"].append(chunk_metadata)
                    filtered_collection["documents"].append(collection["documents"][i])
        else:
            filtered_collection = self.vectorstore.get()
        bm25_retriever = BM25Retriever.from_texts(texts=filtered_collection["documents"],
                                                  metadatas=filtered_collection["metadatas"])
        # maximum number of chunks to retrieve for BM25 retriever
        bm25_retriever.k = self.chunk_k if not self.rerank else self.chunk_k_for_rerank
        # maximum number of chunks to retrieve for vectorstore retriever
        search_kwargs = {"k": self.chunk_k if not self.rerank else self.chunk_k_for_rerank}
        if search_filter is not None:
            # logger.info(f"querying vector store with filter {search_filter}")
            search_kwargs["filter"] = search_filter
        if self.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.score_threshold
        vectorstore_retriever = self.vectorstore.as_retriever(search_type=self.search_type,
                                                              search_kwargs=search_kwargs)
        # Now set EnsembleRetriever for hybrid search
        retriever = EnsembleRetriever(retrievers=[bm25_retriever, vectorstore_retriever],
                                      weights=self.retriever_weights)

        return retriever

    def get_parent_document_retriever(self, search_filter=None):
        """
        returns parent document retriever
        """
        # Use the custom ParentDocumentRetriever that returns the parent docs associated with the child docs
        # maximum number of chunks to retrieve
        search_kwargs = {"k": self.chunk_k_child if not self.rerank else self.chunk_k_for_rerank}
        # filter, if set
        if search_filter is not None:
            # logger.info(f"querying vector store with filter {search_filter}")
            search_kwargs["filter"] = search_filter
        if self.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.score_threshold
        retriever = ParentDocumentRetriever(vectorstore=self.vectorstore,
                                            search_type=self.search_type,
                                            search_kwargs=search_kwargs,
                                            chunk_k=self.chunk_k)

        return retriever

    def get_compression_retriever(self, search_filter=None):
        """
        returns compression retriever to allow reranking of the retrieved documents
        """
        # maximum number of chunks to retrieve is restricted to self.chunk_k_for_rerank
        base_retriever = self.get_vectorstore_retriever(search_filter)
        if self.retriever_type == "hybrid":
            base_retriever = self.get_ensemble_retriever(search_filter)
        elif self.retriever_type == "parent":
            base_retriever = self.get_parent_document_retriever(search_filter)

        compressor = None
        if self.rerank_provider == "flashrank_rerank":
            my_client = Ranker(model_name=self.rerank_model, cache_dir="flashrank_models")
            # maximum number of chunks to retrieve is now restricted to self.chunk_k
            compressor = FlashrankRerank(client=my_client, top_n=self.chunk_k)

        retriever = ContextualCompressionRetriever(base_retriever=base_retriever,
                                                   base_compressor=compressor)

        return retriever

    def get_retriever(self, search_filter=None):
        """
        returns, based on the RETRIEVER_TYPE settings, the retriever object
        """
        if self.rerank:
            retriever = self.get_compression_retriever(search_filter)
            logger.info(f"Reranking with base_retriever {self.retriever_type} and compressor {self.rerank_provider}")
        else:
            if self.retriever_type == "vectorstore":
                retriever = self.get_vectorstore_retriever(search_filter=search_filter)
            elif self.retriever_type == "hybrid":
                retriever = self.get_ensemble_retriever(search_filter)
            elif self.retriever_type == "parent":
                retriever = self.get_parent_document_retriever(search_filter)

        logger.info(f"Set retriever to {self.retriever_type}")

        if self.multiquery:
            # the llm to create multiple questions from the user question
            llm = ChatOpenAI(temperature=0)
            # multiqueryretriever creates 3 queries iso 1
            retriever = MultiQueryRetriever.from_llm(retriever=retriever,
                                                     llm=llm,
                                                     include_original=True)
            logger.info("Using multiple queries")

        return retriever
