from langchain_community.vectorstores.chroma import Chroma
# local imports
import settings_template as settings


class VectorStoreCreator():
    """
    VectorStore class to import into other modules
    """
    def __init__(self, vecdbtype=None) -> None:
        self.vecdb_type = settings.VECDB_TYPE if vecdbtype is None else vecdbtype

    def get_vectorstore(self, embeddings, input_folder, vecdb_folder):
        """ Creates a vector store object given the input folder, embeddings and the folder to persist the database

        Parameters
        ----------
        collection_name : str
            name of the collection to create
        embeddings : Embeddings
            LangChain embeddings from the chosen embedding model
        vectordb_folder : str
            the name of the persist folder

        Returns
        -------
        Chroma
            Chroma vector database object
        """
        if self.vecdb_type == "chromadb":
            vectorstore = Chroma(
                collection_name=input_folder,
                embedding_function=embeddings,
                persist_directory=vecdb_folder,
                collection_metadata={"hnsw:space": "cosine"}
                )

        return vectorstore
