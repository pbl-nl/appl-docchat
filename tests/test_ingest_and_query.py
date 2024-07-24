'''Unit testing for Document Ingestion'''

# glboal imports
import unittest
import os
import sys 
import shutil  

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))) # Add the root directory to the system path
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut


# Test case for the function
class TestIngester(unittest.TestCase):
    '''test the ingestion of different modules'''

    def test_openai(self):
        """Test if the sample_function runs without raising any errors."""
        try:
                # define model type
            embeddings_provider = "openai"
            embeddings_model = "text-embedding-ada-002"

            content_folder_name = 'unit_test_openai'
            content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
            # delete a vector base if one is in place
            if os.path.exists(vectordb_folder_path):
                shutil.rmtree(vectordb_folder_path)
            print(vectordb_folder_path)
            self.ingester = Ingester(collection_name='UT_openai_' + content_folder_name, 
                                content_folder=content_folder_path, 
                                vecdb_folder=vectordb_folder_path,
                                embeddings_provider=embeddings_provider,
                                embeddings_model=embeddings_model)
            self.ingester.ingest()
            self.assertEqual(None, None)
            del self.ingester
        except Exception as e:
            self.fail(f"The function raised an error: {e}")



class TestQuerier(unittest.TestCase):
    '''test the query of different providers'''

    def test_openai(self):
        llm_type = "chatopenai"
        llm_model_type = "gpt-3.5-turbo"
        content_folder_name = 'unit_test_openai'
        _, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
        querier = Querier(llm_type=llm_type, llm_model_type=llm_model_type)
        querier.make_chain(content_folder_name, vectordb_folder_path)
        _ = querier.ask_question('What is her education?')
        self.assertEqual(None, None)

if __name__ == '__main__':
    unittest.main()


