from enum import Enum
from typing import List
import langchain.docstore.document as docstore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class SearchType(str, Enum):
    """
    Enumerator of the types of search to perform.
    """
    # Similarity search
    SIMILARITY = "similarity"
    # Similarity search with threshold
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
    # Maximal Marginal Relevance reranking of similarity search
    MMR = "mmr"


class ParentDocumentRetriever(BaseRetriever):
    """
    Custom ParentDocumentRetriever using just the vectorstore, no file storage needed
    Parent chunks are added to the child chunks as metadata
    """
    # The underlying vectorstore to use to store small chunks
    vectorstore: VectorStore
    # Keyword arguments to pass to the search function
    search_kwargs: dict = Field(default_factory=dict)
    # Type of search to perform (similarity / similarity_score_threshold / mmr)
    search_type: SearchType = SearchType.SIMILARITY

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get documents relevant to a query.

        Parameters
        ----------
        query : str
            String to find relevant documents for
        run_manager : CallbackManagerForRetrieverRun
            The callbacks handler to use

        Returns
        -------
        List[Document]
            List of relevant documents

        Raises
        ------
        ValueError
            _description_
        """
        if self.search_type == SearchType.MMR:
            child_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == SearchType.SIMILARITY:
            child_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
            docs_and_similarities = self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
            child_docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(
                f"search_type of {self.search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold' or 'mmr'"
            )

        # get unique parent docs from child docs metadata
        parent_docs = []
        parent_chunk_ids = []
        for child_doc in child_docs:
            parent_chunk = child_doc.metadata['parent_chunk']
            parent_chunk_id = child_doc.metadata['parent_chunk_id']
            parent_doc = docstore.Document(
                 page_content=parent_chunk,
                 metadata=child_doc.metadata,
                 )
            if parent_chunk_id not in parent_chunk_ids:
                parent_chunk_ids.append(parent_chunk_id)
                parent_docs.append(parent_doc)

        # return parent docs
        # SPT restrict to maximally chunk_k!
        return [parent_doc for parent_doc in parent_docs if parent_doc is not None]
