
from typing import List
import os
import pandas as pd
from loguru import logger
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut


def ingest_or_load_documents(content_folder_name: str, content_folder_path: str, vectordb_folder_path: str) -> None:
    '''
    Depending on whether the vector store already exists, files will be chunked and stored in vectorstore or not
    '''
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vectordb_folder_path):
        # ingest documents
        ingester = Ingester(content_folder_name, content_folder_path, vectordb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")


def get_review_questions(question_list_path: str) -> List[str]:
    '''
    Convert the file with questions into a list of questions
    '''
    # Get questions from question list file
    with open(file=question_list_path, mode='r', encoding="utf8") as review_file:
        review_questions = []
        # read each line
        for line in review_file:
            # add question to list
            review_questions.append(line.strip("\n"))
    return review_questions


def generate_answer(querier: Querier, review_question: str):
    '''
    Generate an answer to the given question with the provided Querier instance
    '''
    # Iterate over the questions and generate the answers
    querier.clear_history()
    response, _ = querier.ask_question(review_question)
    return response["answer"], response["source_documents"]


def main() -> None:
    '''
    Main loop of this module
    '''
    # Get source folder with papers from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated vectordb path
    content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
    review_files = os.listdir(content_folder_path)
    question_list_path = os.path.join(content_folder_path, "review", "questions.txt")

    # If vector store folder does not exist, stop
    if not os.path.exists(content_folder_path):
        logger.info("This content folder does not exist. Please make sure the spelling is correct")
        ut.exit_program()
    elif not os.path.exists(question_list_path):
        logger.info("This question list does not exist, please make sure this list exists.")
        ut.exit_program()

    # Create instance of Querier once
    querier = Querier()

    # ingest documents if documents in source folder path are not ingested yet
    ingest_or_load_documents(content_folder_name, content_folder_path, vectordb_folder_path)

    # Get review questions from file
    review_questions = get_review_questions(question_list_path)

    # create empty dataframe
    df_result = pd.DataFrame(columns=["filename", "question", "answer", "sources"])

    # create the query chain with a search filter and answer each question for each paper
    cntrow = 0
    for review_file in review_files:
        if os.path.isfile(os.path.join(content_folder_path, review_file)):
            logger.info(f"current file: {review_file}")
            for review_question in review_questions:
                cntrow += 1
                querier.make_chain(content_folder_name, vectordb_folder_path, search_filter={"filename": review_file})
                # Generate answer
                answer, sources = generate_answer(querier, review_question)
                df_result.loc[cntrow] = [review_file, review_question, answer, sources]
    output_path = os.path.join(content_folder_path, "review", "result.tsv")
    # sort on question, then on paper
    df_result = df_result.sort_values(by=["question", "filename"])
    df_result.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
