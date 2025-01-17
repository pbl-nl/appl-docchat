'''Unit testing for Document Ingestion'''

# global imports
import unittest
import os
import sys 
import shutil  
from pathlib import Path
import requests

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut
import settings
# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# Test case for the function
class TestIngester(unittest.TestCase):
    '''test the ingestion of different modules'''

    def test_debugging(self):
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        assert endpoint is not None, "AZURE_OPENAI_ENDPOINT is not set"
        assert api_key is not None, "AZURE_OPENAI_API_KEY is not set"
        print("Endpoint:", endpoint)
        print("API Key Length:", len(api_key))

    def test_internet_connection(self):
        """Test if the internet connection is available."""
        try:
            response = requests.get("https://www.google.com", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.ConnectionError:
            self.fail("No internet connection available.")


    def test_openai_ingest(self):
        """Test if the sample_function runs without raising any errors."""
        # define model type
        embeddings_provider = "azureopenai"
        embeddings_model = "text-embedding-ada-002"


        content_folder_path = 'C:/Users/krugerc/OneDrive - Planbureau voor de Leefomgeving/Bureaublad/GitReps/appl-docchat/docs/unit_test_openai'
        content_folder_name = os.path.basename(content_folder_path)
        vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                embeddings_model=embeddings_model)
        # create subfolder for storage of vector databases if not existing
        ut.create_vectordb_folder(content_folder_path)
        # store documents in vector database if necessary
        self.ingester = Ingester(collection_name=content_folder_name,
                            content_folder=content_folder_path,
                            vecdb_folder=vecdb_folder_path,
                            embeddings_provider=embeddings_provider,
                            embeddings_model=embeddings_model)
        self.ingester.ingest()
        self.assertEqual(None, None)



class TestQuerier(unittest.TestCase):
    '''test the query of different providers'''

    def test_openai_query(self):
        llm_provider = "azureopenai"
        llm_model = "gpt-4"
        embeddings_provider = "azureopenai"
        embeddings_model = "text-embedding-ada-002"
        content_folder_path =  'C:/Users/krugerc/OneDrive - Planbureau voor de Leefomgeving/Bureaublad/GitReps/appl-docchat/docs/unit_test_openai'
        content_folder_name = os.path.basename(content_folder_path)
        querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

        # get associated vectordb path
        vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                embeddings_model=embeddings_model)
        querier.make_chain(content_folder_name, vecdb_folder_path)
        question = 'What is her education?'
        response = querier.ask_question(question)
        # _, vectordb_folder_path = ut.create_vectordb_path(content_folder_name)
        # querier = Querier(llm_provider=llm_provider, llm_model=llm_model)
        # querier.make_chain(content_folder_name, vectordb_folder_path)
        # _ = querier.ask_question('What is her education?')
        self.assertEqual(None, None)


if __name__ == '__main__':
    unittest.main()


