'''Unit testing for Document Ingestion'''
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut
# other imports 
import unittest
import os
import shutil      


# Test case for the function
class TestIngester(unittest.TestCase):
    '''test the ingestion of different modules'''

    def test_openai(self):
        """Test if the sample_function runs without raising any errors."""
        try:
                # define model type
            embeddings_provider = "openai"
            embeddings_model = "text-embedding-ada-002"

            content_folder_name = 'unit_test_openai/'
            content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
            # delete a vector base if one is in place
            if os.path.exists(vectordb_folder_path):
                shutil.rmtree(vectordb_folder_path)
            print(vectordb_folder_path)
            self.ingester = Ingester(collection_name='UT_openai_' + content_folder_name, 
                                content_folder=content_folder_path, 
                                vectordb_folder=vectordb_folder_path,
                                embeddings_provider=embeddings_provider,
                                embeddings_model=embeddings_model)
            result = self.ingester.ingest()
            self.assertEqual(result, None)
            del self.ingester, result
        except Exception as e:
            self.fail(f"The function raised an error: {e}")

    def test_huggingface(self):
        """Test if the sample_function runs without raising any errors."""
        try:
                # define model type
            embeddings_provider = "huggingface"
            embeddings_model = "all-mpnet-base-v2"

            content_folder_name = 'unit_test_huggingface/'
            content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
            # delete vector base if it exists
            if os.path.exists(vectordb_folder_path):
                shutil.rmtree(vectordb_folder_path)

            self.ingester = Ingester(collection_name='UT_huggingface_' +content_folder_name, 
                                content_folder=content_folder_path, 
                                vectordb_folder=vectordb_folder_path,
                                embeddings_provider=embeddings_provider,
                                embeddings_model=embeddings_model)
            result = self.ingester.ingest()
            self.assertEqual(result, None)
            del self.ingester, result
            # delete vector base if it exists
        except Exception as e:
            self.fail(f"The function raised an error: {e}")



class TestQuery(unittest.TestCase):
    '''test the query of different providers'''

    def test_openai(self):
        llm_type = "chatopenai"
        llm_model_type = "gpt-3.5-turbo"
        content_folder_name = 'unit_test_openai/'
        _, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
        querier = Querier(llm_type=llm_type, llm_model_type=llm_model_type)
        querier.make_chain(content_folder_name, vectordb_folder_path)
        response, scores = querier.ask_question('What is her education?')
        self.assertEqual(None, None)

    def test_huggingface(self):
        llm_type = "huggingface"
        llm_model_type = "meta-llama/Llama-2-7b-chat-hf"
        content_folder_name = 'unit_test_openai/'
        _, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
        querier = Querier(llm_type=llm_type, llm_model_type=llm_model_type)
        querier.make_chain(content_folder_name, vectordb_folder_path)
        response, scores = querier.ask_question('What is her education?')
        self.assertEqual(None, None)




