from typing import List, Tuple
import os
import csv
import pandas as pd
from loguru import logger
from langchain_core.prompts import PromptTemplate
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut
import prompts.prompt_templates as pr


def ingest_or_load_documents(content_folder_name: str,
                             content_folder_path: str,
                             vectordb_folder_path: str) -> None:
    """
    Depending on whether the vector store already exists, files will be chunked and stored in vectorstore or not

    :param content_folder_name: the name of the folder with content
    :param content_folder_path: the full path of the folder with content
    :param vectordb_folder_path: the full path of the folder with the vector stores
    """
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vectordb_folder_path):
        # ingest documents
        ingester = Ingester(content_folder_name, content_folder_path, vectordb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")


def get_review_questions(question_list_path: str) -> List[Tuple[int, str, str]]:
    """
    Convert the file with questions into a list of questions

    :param question_list_path: the full path of the location containing the file with questions
    :return: list of tuples containing the question id, question type and question
    """
    # Get questions from question list file
    with open(file=question_list_path, mode='r', encoding="utf8") as review_file:
        review_questions = []
        # read each line
        cntline = 0
        for line in review_file:
            elements = line.strip().split("\t")
            # Ignore header, add question to list
            if cntline > 0:
                review_questions.append((cntline, elements[0], elements[1]))
            cntline += 1
    return review_questions


def generate_answer(querier: Querier,
                    review_question: Tuple[int, str, str]) -> Tuple[str, str]:
    """
    Generate an answer to the given question with the provided Querier instance

    :param querier: the Querier object
    :param review_question: the question that is being asked
    :return: tuple containing the answer and the associated sources used (in string form)
    """
    # Generate the answer for the question
    if review_question[1].lower() == "initial":
        querier.clear_history()
    response = querier.ask_question(review_question[2])
    source_docs = ""
    for doc in response["source_documents"]:
        source_docs += f"page {str(doc.metadata['page_number'])}\n{doc.page_content}\n\n"
    return response["answer"], source_docs


def create_answers_for_folder(review_files: list,
                              review_questions: List[Tuple[int, str, str]],
                              content_folder_path: os.PathLike,
                              content_folder_name: str,
                              querier: Querier,
                              vectordb_folder_path: str,
                              output_path: os.PathLike) -> None:
    """
    phase 1 of the review: loop over all the questions, gather the answers and store on disk
    """
    # create empty dataframe
    df_result = pd.DataFrame(columns=["filename", "question_id", "question_type", "question", "answer", "sources"])
    cntrow = 0
    for review_file in review_files:
        if os.path.isfile(os.path.join(content_folder_path, review_file)):
            # create the query chain with a search filter and answer each question for each paper
            querier.make_chain(content_folder_name, vectordb_folder_path, search_filter={"filename": review_file})
            metadata = querier.get_meta_data_by_file_name(review_file)
            for review_question in review_questions:
                logger.info(f"reviewing question {review_question[0]} for file: {review_file}")
                cntrow += 1
                # Generate answer
                answer, sources = generate_answer(querier, review_question)
                answer_plus_name = f"This answer is from {metadata['filename']}:\n {answer}"
                df_result.loc[cntrow] = [review_file,
                                         review_question[0],
                                         review_question[1],
                                         review_question[2],
                                         answer_plus_name,
                                         sources]
    # sort on question, then on document
    df_result = df_result.sort_values(by=["question_id", "filename"])
    df_result.to_csv(output_path, sep="\t", index=False)


def synthesize_results(querier: Querier,
                       results_path: str,
                       output_path: str) -> None:
    """
    phase 2 of the review: synthesizes, per question, the results from phase 1
    """
    # load questions and answers
    answers_df = pd.read_csv(results_path, delimiter='\t')
    # loop over questions
    result = {}
    for question_num in answers_df["question_id"].unique():
        logger.info(f"synthesizing answers for question {question_num}")
        # put all answers to the question in one string
        df_specific_questions = answers_df.loc[answers_df["question_id"] == question_num].copy()
        # make sure that the LLM understands a new paper starts
        answer_list = [question for question in df_specific_questions['answer']]
        answer_string = "\n\n\n\n New Paper:\n".join(answer_list)
        question = df_specific_questions["question"].iloc[0]
        # synthesize results: load prompt for synthesis
        synthesize_prompt_template = PromptTemplate.from_template(template=pr.SYNTHESIZE_PROMPT_TEMPLATE)
        synthesis_prompt = synthesize_prompt_template.format(question=question, answer_string=answer_string)
        synthesis = querier.llm.invoke(synthesis_prompt)
        result[question] = synthesis.content
    # write results
    with open(file=output_path, mode='w', newline='', encoding="utf8") as file:
        # create a writer object specifying TAB as delimiter
        tsv_writer = csv.writer(file, delimiter='\t')
        # write the header
        tsv_writer.writerow(['question', 'answer'])
        # write data
        for key, value in result.items():
            tsv_writer.writerow([key, value])


def main() -> None:
    """
    Main loop of this module
    """
    # get source folder with papers from user
    content_folder_name = input("Source folder of documents (without path): ")
    # get associated vectordb path
    content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
    review_files = os.listdir(content_folder_path)
    question_list_path = os.path.join(content_folder_path, "review", "questions.txt")

    # if vector store folder does not exist, stop
    if not os.path.exists(content_folder_path):
        logger.info("This content folder does not exist. Please make sure the spelling is correct")
        ut.exit_program()
    elif not os.path.exists(question_list_path):
        logger.info(f"This question list does not exist, please make sure this list exists at {question_list_path}.")
        ut.exit_program()

    # create instance of Querier once
    querier = Querier()

    # ingest documents if documents in source folder path are not ingested yet
    ingest_or_load_documents(content_folder_name, content_folder_path, vectordb_folder_path)

    # get review questions from file
    review_questions = get_review_questions(question_list_path)
    # check if there is already a result, if so skip creation of answers
    output_path_review = os.path.join(content_folder_path, "review", "result.tsv")
    if os.path.exists(output_path_review):
        logger.info("A review result (result.tsv) file already exists, skipping the answer creation")
    else:
        create_answers_for_folder(review_files, review_questions, content_folder_path, content_folder_name,
                                  querier, vectordb_folder_path, output_path_review)
        logger.info("Successfully reviewed the documents.")

    # check if synthsis already exists, if not create one
    output_path_synthesis = os.path.join(content_folder_path, "review", "synthesis.tsv")
    if os.path.exists(output_path_synthesis):
        logger.info("A synthesis result file (synthesis.tsv) already exists, skipping the answer creation")
    else:
        # second phase: synthesize the results
        synthesize_results(querier, output_path_review, output_path_synthesis)
        logger.info("Successfully synthesized results.")


if __name__ == "__main__":
    main()
