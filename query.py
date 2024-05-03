import os
from loguru import logger
# local imports
from query.querier import Querier
import utils as ut


def main():
    # create instance of Querier once
    querier = Querier()
    # get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated vectordb path
    _, vecdb_folder_path = ut.create_vectordb_name(content_folder_name)

    # if vector store folder does not exist, stop
    if not os.path.exists(vecdb_folder_path):
        logger.info("There is no vector database for this folder yet. First run \"python ingest.py\"")
        ut.exit_program()
    else:
        # else create the query chain
        querier.make_chain(content_folder_name, vecdb_folder_path)
        while True:
            # get question from user
            question = input("Question: ")
            if question not in ["exit", "quit", "q"]:
                # generate answer and include sources used to produce that answer
                response = querier.ask_question(question)
                logger.info(f"\nAnswer: {response['answer']}")
                # if the retriever returns one or more chunks with a score above the threshold
                if len(response["source_documents"]) > 0:
                    # log the answer to the question and the sources used for creating the answer
                    logger.info("\nSources:\n")
                    for document in response["source_documents"]:
                        logger.info(f"Page {document.metadata['page_number']}, chunk text: {document.page_content}\n")
            else:
                ut.exit_program()


if __name__ == "__main__":
    main()
