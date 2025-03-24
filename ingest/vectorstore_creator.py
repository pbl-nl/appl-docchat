"""
VectorStore class to be used in other modules
"""
# imports
from chromadb.config import Settings
from langchain_chroma import Chroma


class VectorStoreCreator():
    """
    VectorStore class to import into other modules
    """
    def get_vectorstore(self, embeddings, content_folder, vecdb_folder):
        """ Creates a Chroma vectorstore based on the documents in the content folder
        The collection name will be the name of the document folder

        Parameters
        ----------
        embeddings : Embeddings
            LangChain embeddings from the chosen embedding model
        content_folder : str
            name of the content folder
        vecdb_folder : str
            the name of the persist folder where the vector database is stored

        Returns
        -------
        Chroma
            Chroma vector database object
        """
        # if content_folder contains whitespaces, replace them with underscores
        content_folder = content_folder.replace(" ", "_")

        client_settings = Settings(
            anonymized_telemetry=False
        )

        vectorstore = Chroma(
            collection_name=content_folder,
            embedding_function=embeddings,
            persist_directory=vecdb_folder,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=client_settings
        )

        return vectorstore
