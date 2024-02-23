import os
import json
import csv
from dotenv import load_dotenv
from loguru import logger
# local imports
import settings
# from ingest.content_iterator import ContentIterator
from ingest.ingest_utils import IngestUtils
from ingest.file_parser import FileParser
from llm_class.llm_class import LLM


class AutomatedSystematicReview:
    '''
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
    '''
    def __init__(self, content_folder: str, question_list_path: str,
                 embeddings_provider=None, embeddings_model=None, text_splitter_method=None,
                 vecdb_type=None, chunk_size=None, chunk_overlap=None, local_api_url=None,
                 file_no=None, llm_type=None, llm_model_type=None):
        load_dotenv()
        self.content_folder = content_folder
        self.question_list_path = question_list_path
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.API_URL \
            if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type

        # define llm
        self.llm = LLM(self.llm_type, self.llm_model_type).get_llm()

    def conduct_review(self) -> None:
        '''
        Creates file parser object and ingestutils object and iterates over all files in the folder
        '''
        file_parser = FileParser()
        ingestutils = IngestUtils(self.chunk_size, self.chunk_overlap, self.file_no, self.text_splitter_method)
        # get documents and questions
        files_in_folder = os.listdir(self.content_folder)
        question_list = _get_questions(self.question_list_path)
        with open(file='summary_instructions.txt', mode='r', encoding="utf8") as f:
            summary_instructions = f.read()
        # loop over documents
        for file in files_in_folder:
            # loop over question list
            answer_dict = {}
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
                for _, text in enumerate(documents):
                    question_text = f'''
{summary_instructions}
Question: {question}
Text: {text}
'''
                    answer_json_str = self.llm.invoke(question_text)
                    answer_json_str = str(answer_json_str).strip("content='") # .replace('\\n', '\n')
                    try:
                        answer_json = json.loads(answer_json_str) \
                            if type(json.loads(answer_json_str)) != str else eval(json.loads(answer_json_str))
                    except Exception as e:  
                        logger.info(f'Could not turn the text {answer_json_str} to JSON because of {e}')
                        continue
                    if answer_json['answer_in_text'].lower() == 'true':
                        answer_dict[question] = answer_json['answer']
                        flag = True
                        break
                if not flag:
                    answer_dict[question] = 'Answer not found in text'
            log_to_tsv(answer_dict, file_name=os.path.join(self.content_folder, "review", "ASR.tsv"))


def _get_questions(question_list_path: str, delimiter='\n-') -> list[str]:
    '''loads the question list from the location and returns it as a list of qs'''
    with open(file=question_list_path, mode='r', encoding="utf8") as f:
        text = f.read()
    return text.split(delimiter)


def log_to_tsv(input_dict: dict, file_name: str) -> None:
    """
    Appends a dictionary to the end of a TSV file.

    :param input_dict: Dictionary to append. Keys should match TSV columns.
    :param file_name: Path to the TSV file.
    """
    with open(file_name, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=input_dict.keys(), delimiter='\t')

        # Check if the file is empty to decide whether to write the header
        file.seek(0)
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(input_dict)
