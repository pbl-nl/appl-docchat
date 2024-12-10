"""
Summarization is implemented in 2 ways:
- Quick summarization ("map reduce" method)
- Extensive summarization ("refine" method)
See also: https://python.langchain.com/v0.1/docs/use_cases/summarization/
"""
import os
from loguru import logger
# local imports
from summarize.summarizer import Summarizer
import utils as ut
import settings


def main():
    """
    Main function enabling summarization
    """
    # Get source folder with docs from user
    content_folder_path = input("Source folder path of documents (including path): ")
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    # Get private docs indicator from user
    confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    confidential = confidential_yn in ["y", "Y"]
    # get relevant models
    llm_provider, llm_model, _, embeddings_model = ut.get_relevant_models(confidential)
    # # get associated content folder path and vecdb path
    # content_folder_path, _ = ut.create_vectordb_path(content_folder_path=content_folder_path,
    #                                                  embeddings_model=embeddings_model)
    # choose way of summarizing
    summarization_method = input("Summarization Method [map_reduce (m), refine (r)]: ")
    if summarization_method not in ["map_reduce", "refine", "m", "r"]:
        logger.info("Exiting because of invalid user input, please choose map_reduce or refine")
        ut.exit_program()
    else:
        if summarization_method == "r":
            summarization_method = "refine"
        elif summarization_method == "m":
            summarization_method = "map_reduce"
        # create subfolder for storage of summaries if not existing
        ut.create_summaries_folder(content_folder_path)
        # content_folder_path, _ = ut.create_vectordb_path(content_folder_name)
        summarizer = Summarizer(content_folder_path=content_folder_path,
                                summarization_method=summarization_method,
                                text_splitter_method=settings.SUMMARY_TEXT_SPLITTER_METHOD,
                                chunk_size=settings.SUMMARY_CHUNK_SIZE,
                                chunk_overlap=settings.SUMMARY_CHUNK_OVERLAP,
                                llm_provider=settings.SUMMARY_LLM_PROVIDER,
                                llm_model=settings.SUMMARY_LLM_MODEL)
        logger.info(f"Starting summarizer with method {summarization_method}")
        summarizer.summarize_folder()
        logger.info(f"{content_folder_name} successfully summarized.")


if __name__ == "__main__":
    main()
