"""
Summarization is implemented in 2 ways:
- Quick summarization ("map reduce" method)
- Extensive summarization ("refine" method)
See also: https://python.langchain.com/v0.1/docs/use_cases/summarization/
"""
# imports
import os
from loguru import logger
# local imports
from ingest.ingester import Ingester
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
    # choose way of summarizing
    summarization_method = input("Summarization Method [map_reduce (m), refine (r)]: ")
    if summarization_method not in ["map_reduce", "refine", "m", "r"]:
        logger.info("Exiting because of invalid user input, please choose map_reduce (m) or refine (r)")
        ut.exit_program()
    else:
        # get relevant models
        llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=True,
                                                                                                private=confidential)
        # get associated source folder path and vectordb path
        summary_vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                            retriever_type="vectorstore",
                                                            embeddings_provider=embeddings_provider,
                                                            embeddings_model=embeddings_model,
                                                            text_splitter_method=settings.SUMMARY_TEXT_SPLITTER_METHOD,
                                                            chunk_size=settings.SUMMARY_CHUNK_SIZE,
                                                            chunk_overlap=settings.SUMMARY_CHUNK_OVERLAP)
        # create subfolder for storage of vector databases if not existing
        ut.create_vectordb_folder(content_folder_path)
        # store documents in vector database if necessary, according to summary settings
        ingester = Ingester(collection_name=content_folder_name,
                            content_folder=content_folder_path,
                            vecdb_folder=summary_vecdb_folder_path,
                            embeddings_provider=embeddings_provider,
                            embeddings_model=embeddings_model,
                            retriever_type="vectorstore",
                            text_splitter_method=settings.SUMMARY_TEXT_SPLITTER_METHOD,
                            chunk_size=settings.SUMMARY_CHUNK_SIZE,
                            chunk_overlap=settings.SUMMARY_CHUNK_OVERLAP)
        # ingest any documents if necessary
        ingester.ingest()

        if summarization_method == "r":
            summarization_method = "refine"
        elif summarization_method == "m":
            summarization_method = "map_reduce"
        # create subfolder for storage of summaries if not existing
        ut.create_summaries_folder(content_folder_path)
        # content_folder_path, _ = ut.create_vectordb_path(content_folder_name)
        summarizer = Summarizer(content_folder_path=content_folder_path,
                                vecdb_folder=summary_vecdb_folder_path,
                                summarization_method=summarization_method,
                                embeddings_provider=embeddings_provider,
                                embeddings_model=embeddings_model,
                                llm_provider=llm_provider,
                                llm_model=llm_model)
        logger.info(f"Starting summarizer with method {summarization_method}")
        summarizer.summarize_folder()
        logger.info(f"{content_folder_name} successfully summarized.")


if __name__ == "__main__":
    main()
