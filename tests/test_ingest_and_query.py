'''Unit testing for Document Ingestion'''

# global imports
import unittest
import os
import sys 
import shutil  
from pathlib import Path

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut
# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# Test case for the function
class TestIngester(unittest.TestCase):
    '''test the ingestion of different modules'''

    def test_openai_ingest(self):
        """Test if the sample_function runs without raising any errors."""
        # define model type
        embeddings_provider = "openai"
        embeddings_model = "text-embedding-ada-002"

        content_folder_name = 'unit_test_openai'
        content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
        # delete a vector base if one is in place
        # if os.path.exists(vectordb_folder_path):
        #     shutil.rmtree(vectordb_folder_path)
        print(vectordb_folder_path)
        self.ingester = Ingester(collection_name='UT_openai_' + content_folder_name,
                                    content_folder=content_folder_path,
                                    vecdb_folder=vectordb_folder_path,
                                    embeddings_provider=embeddings_provider,
                                    embeddings_model=embeddings_model)
        self.ingester.ingest()
        self.assertEqual(None, None)
        del self.ingester


class TestQuerier(unittest.TestCase):
    '''test the query of different providers'''

    def test_openai_query(self):
        llm_provider = "openai"
        llm_model = "gpt-3.5-turbo"
        content_folder_name = 'unit_test_openai'
        _, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
        querier = Querier(llm_provider=llm_provider, llm_model=llm_model)
        querier.make_chain(content_folder_name, vectordb_folder_path)
        _ = querier.ask_question('What is her education?')
        self.assertEqual(None, None)


if __name__ == '__main__':
    unittest.main()


