import os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# local imports
from query.llm_creator import LLMCreator
from ingest.splitter_creator import SplitterCreator
from ingest.file_parser import FileParser
import settings
import utils as ut
import prompts.prompt_templates as pr


class Summarizer:
    """
    When the summarizer class parameters are read from settings.py, object is initiated without parameter settings
    When parameters are read from GUI, object is initiated with parameter settings listed
    """
    def __init__(self, content_folder_path: str, summarization_method: str, text_splitter_method=None,
                 chunk_size=None, chunk_overlap=None, llm_provider=None, llm_model=None, in_memory=False) -> None:
        """
        When parameters are read from settings.py, object is initiated without parameter settings
        When parameters are read from GUI, object is initiated with parameter settings listed
        """
        self.content_folder_path = content_folder_path
        self.chain_type = summarization_method
        self.text_splitter_method = settings.SUMMARY_TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.chunk_size = settings.SUMMARY_CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.SUMMARY_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.llm_provider = settings.SUMMARY_LLM_PROVIDER if llm_provider is None else llm_provider
        self.llm_model = settings.SUMMARY_LLM_MODEL if llm_model is None else llm_model
        self.in_memory = in_memory

        # create llm object
        load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
        self.llm = LLMCreator(llm_provider=self.llm_provider,
                              llm_model=self.llm_model).get_llm()

    def make_chain(self, my_language):
        """
        defines the load_summarize_chain to use, depending on method chosen
        """
        partial_map_reduce_prompt = f"Write a concise summary of the following in the {my_language} language: "
        partial_refine_prompt = f"Given the new context, refine the original summary in the {my_language} language. \
        If the context isn't useful, return the original summary."

        if self.chain_type == "map_reduce":
            map_reduce_prompt = PromptTemplate(template=partial_map_reduce_prompt + pr.SUMMARY_PROMPT_TEMPLATE,
                                               input_variables=["text"])
            kwargs = {
                'map_prompt': map_reduce_prompt,
                'combine_prompt': map_reduce_prompt
            }
        else:
            map_reduce_prompt = PromptTemplate(template=partial_map_reduce_prompt + pr.SUMMARY_PROMPT_TEMPLATE,
                                               input_variables=["text"])
            refine_prompt = PromptTemplate(template=pr.SUMMARY_REFINE_TEMPLATE + partial_refine_prompt,
                                           input_variables=["existing_answer", "text"])
            kwargs = {
                "question_prompt": map_reduce_prompt,
                "refine_prompt": refine_prompt
            }

        return load_summarize_chain(llm=self.llm, chain_type=self.chain_type, **kwargs)

    def summarize_folder(self) -> None:
        """
        Create summaries of all files in the folder, using the chosen summarization method.
        One summary per file is created when the summary of the file does not exist yet.
        """
        # create subfolder "summaries" if not existing
        if 'summaries' not in os.listdir(self.content_folder_path):
            os.mkdir(os.path.join(self.content_folder_path, "summaries"))

        # list of relevant files to summarize
        files_in_folder = ut.get_relevant_files_in_folder(self.content_folder_path)

        # loop over all files in the folder
        if self.in_memory:
            # create summaries in memory
            in_memory_summaries = {}
            for file in files_in_folder:
                in_memory_summaries[file] = self.summarize_file(file=file)
            return in_memory_summaries
        # create summaries on disk
        for file in files_in_folder:
            file_name, _ = os.path.splitext(file)
            summary_name = os.path.join(self.content_folder_path, "summaries",
                                        str(file_name) + "_" + str.lower(self.chain_type) + ".txt")
            # if summary does not exist yet, create it
            if not os.path.isfile(summary_name):
                self.summarize_file(file=file)

    def summarize_file(self, file: str) -> None:
        """
        creates summary for one specific file
        """
        # detect language first
        file_parser = FileParser()
        _, metadata = file_parser.parse_file(os.path.join(self.content_folder_path, file), in_memory=self.in_memory)
        language = ut.LANGUAGE_MAP.get(metadata['Language'], 'english')
        # create splitter object
        text_splitter = SplitterCreator(text_splitter_method=self.text_splitter_method,
                                        chunk_size=self.chunk_size,
                                        chunk_overlap=self.chunk_overlap).get_splitter(language)

        loader = PyMuPDFLoader(os.path.join(self.content_folder_path, file))
        docs = loader.load_and_split(text_splitter=text_splitter)
        chain = self.make_chain(language)
        summary = chain.invoke(docs)["output_text"]
        if self.in_memory:
            return summary
        # store summary on disk
        file_name, _ = os.path.splitext(file)
        result = os.path.join(self.content_folder_path, "summaries", str(file_name) + "_" +
                              str.lower(self.chain_type) + ".txt")
        with open(file=result, mode="w", encoding="utf8") as f:
            f.write(summary)
