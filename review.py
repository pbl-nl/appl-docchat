"""
The review module focuses on comparison of documents.
The module is split in two phases:
1. Creation of answers for each question and each document sequentially
2. Synthesis of the answers for each question
"""
from typing import List, Tuple
import os
import csv
from datetime import datetime
import pandas as pd
from loguru import logger
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import utils as ut
import settings


def ingest_or_load_documents(
    content_folder_name: str, content_folder_path: str, vecdb_folder_path: str
) -> None:
    """
    Depending on whether the vector store already exists, files will be chunked and stored in vectorstore or not

    Parameters
    ----------
    content_folder_name : str
        the name of the folder with content
    content_folder_path : str
        the full path of the folder with content
    vecdb_folder_path : str
        the full path of the folder with the vector stores
    """
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vecdb_folder_path):
        # ingest documents
        ingester = Ingester(collection_name=content_folder_name,
                            content_folder=content_folder_path,
                            vecdb_folder=vecdb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vecdb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")


def generate_answer(
    querier: Querier, review_question_type: str, review_question: str
) -> Tuple[str, str]:
    """
    Generate an answer to the given question with the provided Querier instance

    Parameters
    ----------
    querier : Querier
        the Querier object
    review_question :str
        the question to be answered
    review_question_type : str
        the type of the question [initial or follow up]

    Returns
    -------
    Tuple[str, str]]
        tuple containing the answer and the associated sources used (in string form)
    """
    # Generate the answer for the question
    if review_question_type.lower() == "initial":
        querier.clear_history()
    response = querier.ask_question(review_question)
    source_docs = ""
    for doc in response["source_documents"]:
        source_docs += (
            f"page {str(doc.metadata['page_number'])}\n{doc.page_content}\n\n"
        )

    return response["answer"], source_docs


def write_settings(input_path: os.PathLike, confidential: bool, output_path: os.PathLike) -> None:
    """
    Stores relevant settings to the output file, for reproducability purposes

    Parameters
    ----------
    output_path : os.PathLike
        path of the output file
    """
    with open(file=output_path, mode="w", encoding="utf8") as file:
        file.write(f"input path =  {input_path} \n")
        file.write(f"confidential =  {confidential} \n")
        file.write(f"settings.TEXT_SPLITTER_METHOD =  {settings.TEXT_SPLITTER_METHOD} \n")
        file.write(f"settings.TEXT_SPLITTER_CHILD =  {settings.TEXT_SPLITTER_METHOD_CHILD} \n")
        file.write(f"settings.CHUNK_SIZE =  {settings.CHUNK_SIZE} \n")
        file.write(f"settings.CHUNK_SIZE_CHILD =  {settings.CHUNK_SIZE_CHILD} \n")
        file.write(f"settings.CHUNK_K =  {settings.CHUNK_K} \n")
        file.write(f"settings.CHUNK_K_CHILD =  {settings.CHUNK_K_CHILD} \n")
        file.write(f"settings.CHUNK_OVERLAP =  {settings.CHUNK_OVERLAP} \n")
        file.write(f"settings.CHUNK_OVERLAP_CHILD =  {settings.CHUNK_OVERLAP_CHILD} \n")
        if not confidential:
            file.write(f"settings.EMBEDDINGS_PROVIDER =  {settings.EMBEDDINGS_PROVIDER} \n")
            file.write(f"settings.EMBEDDINGS_MODEL =  {settings.EMBEDDINGS_MODEL} \n")
            file.write(f"settings.LLM_PROVIDER =  {settings.LLM_PROVIDER} \n")
            file.write(f"settings.LLM_MODEL =  {settings.LLM_MODEL} \n")
        else:
            file.write(f"settings.PRIVATE_EMBEDDINGS_PROVIDER =  {settings.PRIVATE_EMBEDDINGS_PROVIDER} \n")
            file.write(f"settings.PRIVATE_EMBEDDINGS_MODEL =  {settings.PRIVATE_EMBEDDINGS_MODEL} \n")
            file.write(f"settings.PRIVATE_LLM_PROVIDER =  {settings.PRIVATE_LLM_PROVIDER} \n")
            file.write(f"settings.PRIVATE_LLM_MODEL =  {settings.PRIVATE_LLM_MODEL} \n")
        file.write(f"settings.SEARCH_TYPE =  {settings.SEARCH_TYPE} \n")
        file.write(f"settings.SCORE_THRESHOLD =  {settings.SCORE_THRESHOLD} \n")
        file.write(f"settings.RETRIEVER_TYPE =  {settings.RETRIEVER_TYPE} \n")
        file.write(f"settings.RERANK =  {settings.RERANK} \n")
        file.write(f"settings.RERANK_PROVIDER =  {settings.RERANK_PROVIDER} \n")
        file.write(f"settings.RERANK_MODEL =  {settings.RERANK_MODEL} \n")
        file.write(f"settings.CHUNK_K_FOR_RERANK =  {settings.CHUNK_K_FOR_RERANK} \n")
        file.write(f"settings.RETRIEVER_PROMPT_TEMPLATE =  {settings.RETRIEVER_PROMPT_TEMPLATE} \n\n")


def create_answers_for_folder(question_list_path: str,
                              review_files: List[str],
                              content_folder_name: str,
                              querier: Querier,
                              vecdb_folder_path: str,
                              output_path: os.PathLike) -> None:
    """
    Phase 1 of the review: loop over all the questions and all the documents, gather the answers and store on disk
    Phase 2 of the review (if the file questions.csv contains a value for summary_template)

    Parameters
    ----------
    question_list_path : str
        path of the file with the list of questions
    review_files : List[str]
        list of files to be reviewed
    content_folder_name : str
        name of the document folder
    querier : Querier
        the Querier object
    vecdb_folder_path : str
        path of the vector database
    output_path : os.PathLike
        path of the output file
    """
    # create empty dataframe
    df_result = pd.DataFrame(
        columns=[
            "filename",
            "question_id",
            "question_type",
            "question",
            "answer",
            "assistant_prompt",
            "sources"
        ]
    )
    cntrow = 0

    # load review questions
    review_questions = pd.read_csv(filepath_or_buffer=question_list_path)
    review_questions.dropna(inplace=True, how="all")

    # loop over each combination of question, question template and summary template
    for index, row in review_questions.iterrows():
        # review_question = list(review_question)
        review_question_type = row["Question_Type"]
        review_question = row["Question"]
        review_instruction_template = row["Instruction_Template"]
        review_summary_template = row["summary_template"]
        logger.info(f"reviewing question {review_question}")
        for review_file in review_files:
            # create the query chain with a search filter and answer each question for each document
            querier.make_chain(content_folder=content_folder_name,
                               vecdb_folder=vecdb_folder_path,
                               search_filter={"filename": review_file},
                               qa_template_file_path_or_string=review_instruction_template)

            # Generate answer
            answer, sources = generate_answer(querier=querier,
                                              review_question_type=review_question_type,
                                              review_question=review_question)

            # check if there is a synthesis prompt. If so, add the document reference to the answer
            answer_plus_document_reference = f"This answer is from {review_file}:\n {answer}"
            final_answer = answer_plus_document_reference if str(review_summary_template) != "nan" else answer

            # add resulting answer and input data to dataframe
            cntrow += 1
            df_result.loc[cntrow] = [
                review_file,
                index,
                review_question_type,
                review_question,
                final_answer,
                review_instruction_template,
                sources
            ]

    # function to clean up the newlines in the text columns
    def clean_newlines(text):
        if isinstance(text, str):
            return text.replace('\n', ' ')
        return text

    # if one of the questions requires summarization, create a summary and save it
    if review_questions['summary_template'].notnull().any():
        summary_result = {}
        # get rows that have a summary template
        summary_rows = review_questions[review_questions['summary_template'].notnull()]
        # loop over the neccessary summaries
        for index, row in summary_rows.iterrows():
            # get the relevant qa_prompt_template_path for that question
            synthesize_prompt_template = row['summary_template']
            # get all answers for the question in the dataframe
            answers = df_result[df_result['question_id'] == index]['answer']
            synthesis_prompt = synthesize_prompt_template.format(
                question=row['Question'], answer_string="\n\n".join(answers)
                )
            # query llm for synthesis
            synthesis = querier.llm.invoke(synthesis_prompt)
            summary_result[row['Question']] = synthesis.content
        # write results
        output_path_str = str(output_path)
        filename, extension = os.path.splitext(output_path_str)
        output_path_summary = f"{filename}_summary{extension}"
        with open(file=output_path_summary, mode="a", newline="", encoding="utf8") as file:
            # create a writer object specifying TAB as delimiter
            tsv_writer = csv.writer(file, delimiter="\t")
            # write the header
            tsv_writer.writerow(["question", "answer"])
            # write data
            for key, value in summary_result.items():
                tsv_writer.writerow([key, clean_newlines(value)])

    # Apply to columns that contain text with newlines
    text_columns = ['question', 'answer', 'sources', 'question_type', 'assistant_prompt']  # 'filename' is not included
    for col in text_columns:
        df_result[col] = df_result[col].apply(clean_newlines)

    # Then save to TSV
    df_result = df_result.sort_values(by=["question_id", "filename"])
    df_result.to_csv(output_path, sep='\t', index=False, mode='a')


def main(content_folder_path) -> None:
    """
    Main loop of this module
    """
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    confidential = False
    # get relevant models
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                            private=confidential)
    # get associated content folder path and vecdb path
    vecdb_folder_path = ut.create_vectordb_path(content_folder_path=content_folder_path,
                                                embeddings_provider=embeddings_provider,
                                                embeddings_model=embeddings_model)
    # if content folder path does not exist, stop
    if not os.path.exists(content_folder_path):
        logger.info(
            "This content folder does not exist. Please make sure the spelling is correct"
        )
        ut.exit_program()

    # get path of file with list of questions
    question_list_path = os.path.join(content_folder_path, "review", "questions.csv")

    # if question list path does not exist, stop
    if not os.path.exists(question_list_path):
        logger.info(
            f"The file with questions does not exist, please make sure it exists at {question_list_path}."
        )
        ut.exit_program()

    # get list of relevant files in document folder
    review_files = ut.get_relevant_files_in_folder(content_folder_path)

    # create output folder with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hhour_%Mmin_%Ssec")
    os.mkdir(os.path.join(content_folder_path, f"review/{timestamp}"))
    # copy the question list file to the output folder
    os.system(f"cp {question_list_path} {content_folder_path}/review/{timestamp}/questions.csv")

    # ingest documents if documents in source folder path are not ingested yet
    ingest_or_load_documents(content_folder_name=content_folder_name,
                             content_folder_path=content_folder_path,
                             vecdb_folder_path=vecdb_folder_path)

    # write settings to file
    output_path_settings = os.path.join(
        content_folder_path, f"review/{timestamp}", "settings.txt"
    )
    write_settings(input_path=content_folder_path,
                   confidential=confidential,
                   output_path=output_path_settings)

    # Create answers and store them in the file specified by output_path
    output_path_review = os.path.join(
        content_folder_path, f"review/{timestamp}", "answers.tsv"
    )

    # create instance of Querier once
    querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

    create_answers_for_folder(
        question_list_path=question_list_path,
        review_files=review_files,
        content_folder_name=content_folder_name,
        querier=querier,
        vecdb_folder_path=vecdb_folder_path,
        output_path=output_path_review
    )
    logger.info("Successfully reviewed the documents.")


if __name__ == "__main__":
    # get source folder with papers from user
    content_folder_path = input("Source folder of documents (including path): ")
    main(content_folder_path=content_folder_path)
