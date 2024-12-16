import os
from loguru import logger
# local imports
from query.querier import Querier
import utils as ut


def main():
    # get source folder with docs from user
    content_folder_path = input("Source folder of documents (including path): ")
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    # Get private docs indicator from user
    confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    confidential = confidential_yn in ["y", "Y"]
    # get relevant models
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(confidential)
    # create instance of Querier once
    querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

    # get associated vectordb path
    vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                embeddings_model=embeddings_model)

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
                        logger.info(f"File {document.metadata['filename']}, \
                                    Page {document.metadata['page_number'] + 1}, \
                                    chunk text: {document.page_content}\n")
            else:
                ut.exit_program()


if __name__ == "__main__":
    main()
