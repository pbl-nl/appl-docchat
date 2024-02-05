import os
from dotenv import load_dotenv
from loguru import logger
# local imports
import settings
# from ingest.content_iterator import ContentIterator
from ingest.ingest_utils import IngestUtils
from ingest.file_parser import FileParser
import utils as ut
import json
import csv

from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class AutomatedSystematicReview:
    '''
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
    '''
    def __init__(self, content_folder: str, question_list_name: str, 
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, local_api_url=None,
                 file_no=None, llm_type=None, llm_model_type=None, prompt_instructions_location=None):
        
        load_dotenv()
        self.content_folder = content_folder
        self.question_list_name = question_list_name
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.API_URL if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.prompt_instructions_location = settings.PROMPT_INSTRUCTIONS if prompt_instructions_location is None else prompt_instructions_location

        with open('asr/prompt_instructions/' + self.prompt_instructions_location, 'r') as f:
            self.prompt_instructions = f.read()
        
        # if llm_type is "chatopenai"
        if self.llm_type == "chatopenai":
            # default llm_model_type value is "gpt-3.5-turbo"
            self.llm_model_type = "gpt-3.5-turbo"
            if self.llm_model_type == "gpt35_16":
                self.llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                self.llm_model_type = "gpt-4"
            self.llm = ChatOpenAI(
                client=None,
                model=self.llm_model_type,
                temperature=0,
            )
        # else, if llm_type is "huggingface"
        elif self.llm_type == "huggingface":
            # default value is llama-2, with maximum output length 512
            self.llm_model_type = "meta-llama/Llama-2-7b-chat-hf"
            max_length = 512
            if self.llm_model_type == 'GoogleFlan':
                self.llm_model_type = 'google/flan-t5-base'
                max_length = 512
            self.llm = HuggingFaceHub(repo_id=self.llm_model_type,
                                 model_kwargs={"temperature": 0.1,
                                               "max_length": max_length}
                                )
        # else, if llm_type is "local_llm"
        elif self.llm_type == "local_llm":
            logger.info("Use Local LLM")
            logger.info("Retrieving " + self.llm_model_type)
            if self.local_api_url is not None: # If API URL is defined, use it
                logger.info("Using local api url " + self.local_api_url)
                self.llm = Ollama(
                    model=self.llm_model_type, 
                    base_url=self.local_api_url,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            else:
                self.llm = Ollama(
                    model=self.llm_model_type,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            logger.info("Retrieved " + self.llm_model_type)


    def conduct_review(self) -> None:
        '''
        Creates file parser object and ingestutils object and iterates over all files in the folder
        '''
        file_parser = FileParser()
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        # get documents and questions
        files_in_folder = os.listdir(self.content_folder)
        question_list = _get_questions(self.question_list_name)
        # loop over documents
        for file in files_in_folder:
        # loop over question list
            answer_dict = {'file_name': file.split('.')[0]}
            # get document texts
            file_path = os.path.join(self.content_folder, file)
            _, file_extension = os.path.splitext(file_path)
            if file_extension in [".docx", ".html", ".md", ".pdf", ".txt"]:
                # extract raw text pages and metadata according to file type
                raw_pages, metadata = file_parser.parse_file(file_path)
            else:
                logger.info(f"Skipping ingestion of file {file} because it has extension {file[-4:]}")
                continue
            documents = ingestutils.clean_text_to_docs(raw_pages, metadata)
            # iterate over question list
            for question in question_list:
                # loop over text chunks
                flag = False
                for text_chunk_num, text in enumerate(documents):
                    question_text = f'''
{self.prompt_instructions}
Question: {question}
Text: {text}
'''
                    answer_json_str = self.llm.invoke(question_text)
                    answer_json_str = str(answer_json_str).strip("content='") # .replace('\\n', '\n')
                    try:
                        answer_json = json.loads(answer_json_str) if type(json.loads(answer_json_str)) != str else eval(json.loads(answer_json_str))
                    except Exception as e:  
                        logger.info(f'Could not turn the text {answer_json_str} to JSON because of {e}')
                        continue
                    if answer_json['answer_in_text'].lower() == 'true':
                        answer_dict[question] = answer_json['answer']
                        flag = True
                        break
                if not flag:
                    answer_dict[question] = 'Answer not found in text'
            log_to_tsv(answer_dict, self.content_folder + '/ASR.tsv')
                


def _get_questions(question_list_name: str, delimiter='\n-') -> list[str]:
    '''loads the question list from the location and returns it as a list of qs'''
    with open(question_list_name, 'r') as f:
        text = f.read()
    return text.split(delimiter)


def log_to_tsv(input_dict: dict, file_name: str) -> None:
    """
    Appends a dictionary to the end of a TSV file, ensuring values are added to the correct columns.

    :param input_dict: Dictionary to append. Keys should match existing TSV columns, if any.
    :param file_name: Path to the TSV file.
    """
    # First, determine if the file exists and has a header
    file_exists = os.path.isfile(file_name)
    header_exists = False
    if file_exists:
        with open(file_name, 'r', newline='', encoding='utf-8') as file:
            # Check if there's at least one line for the header
            header_exists = bool(file.readline())

    with open(file_name, 'a', newline='', encoding='utf-8') as file:
        if header_exists:
            # Read the existing header to align the input dictionary
            with open(file_name, 'r', newline='', encoding='utf-8') as file_for_header:
                reader = csv.reader(file_for_header, delimiter='\t')
                existing_header = next(reader)
            
            # Filter input_dict to only include keys that match the existing header
            filtered_dict = {key: input_dict[key] for key in existing_header if key in input_dict}
            writer = csv.DictWriter(file, fieldnames=existing_header, delimiter='\t')
        else:
            # File doesn't exist or is empty, use the input_dict keys as the header
            writer = csv.DictWriter(file, fieldnames=input_dict.keys(), delimiter='\t')
            writer.writeheader()
            filtered_dict = input_dict
        
        writer.writerow(filtered_dict)
