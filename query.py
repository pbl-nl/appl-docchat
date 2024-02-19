import os
from loguru import logger
# local imports
from query.querier import Querier
import utils as ut


def main():
    # Create instance of Querier once
    querier = Querier()
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated vectordb path
    _, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)

    # If vector store folder does not exist, stop
    if not os.path.exists(vectordb_folder_path):
        logger.info("There is no vector database for this folder yet. First run \"python ingest.py\"")
        ut.exit_program()
    else:
        # create the query chain
        querier.make_chain(content_folder_name, vectordb_folder_path)
        while True:
            # Get question from user
            question = input("Question: ")
            if question not in ["exit", "quit", "q"]:
                # log the question
                logger.info(f"\nQuestion: {question}")
                # Generate answer and include sources used to produce that answer
                response, scores = querier.ask_question(question)
                logger.info(f"\nAnswer: {response['answer']}")
                # if the retriever returns one or more chunks with a score above the threshold
                if scores[0] >= querier.score_threshold:
                    # log the answer to the question and the sources used for creating the answer
                    logger.info("\nSources:\n")
                    cnt = 0
                    for document in response["source_documents"]:
                        logger.info(f"score: {scores[cnt]}")
                        cnt += 1
                        logger.info(f"Page {document.metadata['page_number']} chunk used: {document.page_content}\n")
            else:
                ut.exit_program()


if __name__ == "__main__":
    main()
