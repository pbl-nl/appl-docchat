'''Unit testing for Document Ingestion'''
from ingest.ingester import Ingester
import utils



class UnitTesting():
    def __init__(self):
        
        # define model type
        embeddings_provider = "openai"
        embeddings_model = "text-embedding-ada-002"

        content_folder_name = 'maggy_cv'
        content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)

        self.ingester = Ingester(collection_name=content_folder_name, 
                            content_folder=content_folder_path, 
                            vectordb_folder=vectordb_folder_path,
                            embeddings_provider=embeddings_provider,
                            embeddings_model=embeddings_model)
    
    def test(self) -> None:
        self.ingester.ingest()
        try:
            self.ingester.ingest()
        except Exception as e:
            raise KeyError(f'Unit Test Failed because {e}')

