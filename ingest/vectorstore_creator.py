from langchain_community.vectorstores.chroma import Chroma
# local imports
import settings


class VectorStoreCreator():
    """
    VectorStore class to import into other modules
    """
    def __init__(self, vecdbtype=None) -> None:
        self.vecdb_type = settings.VECDB_TYPE if vecdbtype is None else vecdbtype

    def get_vectorstore(self, embeddings, content_folder, vecdb_folder):
        """ Creates a vector database object given the content folder, embeddings

        Parameters
        ----------
        embeddings : Embeddings
            LangChain embeddings from the chosen embedding model
        content_folder : str
            name of the content folder
        vectordb_folder : str
            the name of the persist folder where the vector database is stored

        Returns
        -------
        Chroma
            Chroma vector database object
        """
        if self.vecdb_type == "chromadb":
            vectorstore = Chroma(
                collection_name=content_folder,
                embedding_function=embeddings,
                persist_directory=vecdb_folder,
                collection_metadata={"hnsw:space": "cosine"}
                )

        return vectorstore
