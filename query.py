import os
from loguru import logger
# local imports
from query.querier import Querier
from settings import VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE
import utils


def main():
    proceed = True
    querier = Querier(EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated vectordb path
    _, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)

    # If vector store folder does not exist, stop
    if not os.path.exists(vectordb_folder_path):
        logger.info("There is no vector database for this folder yet. First run \"python ingest.py\"")
        proceed = False
    else:
        # Create instance of Querier once
        # querier = Querier(content_folder_name, vectordb_folder_path, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
        querier.make_chain(content_folder_name, vectordb_folder_path)
        while proceed:
            logger.info("")
            # Get question from user
            question = input("Question: ")
            if question != "exit":
                # Generate answer and include sources used to produce that answer
                answer, source = querier.ask_question(question)

                logger.info(f"\nAnswer: {answer}")
                logger.info("\nSources:\n")
                for document in source:
                    logger.info(f"Page {document.metadata['page_number']} chunk used: {document.page_content}\n")
            else:
                proceed = False


if __name__ == "__main__":
    main()
