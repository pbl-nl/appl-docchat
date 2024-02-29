"""
Summarization is implemented in 2 ways:
- Quick summarization (Map Reduce method)
- More extensive summariation (Refine method)
"""
from loguru import logger
# local imports
from summarize.summarizer import Summarizer
import utils


def main():
    """
    Main function enabling summarization
    """
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # choose way of summarizing
    summarization_method = input("Summarization Method [Map_Reduce, Refine]: ")
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    summarizer = Summarizer(content_folder=content_folder_path,
                            collection_name=content_folder_name,
                            summary_method=summarization_method,
                            vectordb_folder=vectordb_folder_path)
    logger.info(f"Starting summarizer with method {summarization_method}")
    summarizer.summarize()
    logger.info(f"{content_folder_name} successfully summarized.")


if __name__ == "__main__":
    main()
