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
import prompts.prompt_templates as pr
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


def get_review_questions(question_list_path: str) -> List[Tuple[int, str, str]]:
    """
    Convert the file with questions into a list of questions

    Parameters
    ----------
    question_list_path : str
        the full path of the location containing the file with questions

    Returns
    -------
    List[Tuple[int, str, str]]
        list of tuples containing the question id, question type and question
    """
    # Get questions from question list file
    with open(file=question_list_path, mode="r", encoding="utf8") as review_file:
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


def get_synthesis_questions(question_list_path: str) -> List[Tuple[int, str, str]]:
    """
    Convert the file with questions into a list of questions

    Parameters
    ----------
    question_list_path : str
        the full path of the location containing the file with questions

    Returns
    -------
    List[Tuple[int, str, str]]
        list of tuples containing the question id, question type and question
    """
    # Get questions from question list file
    with open(file=question_list_path, mode="r", encoding="utf8") as review_file:
        review_questions = []
        # read each line
        cntline = 0
        for line in review_file:
            elements = line.strip().split("\t")
            # Ignore header, add question to list
            if cntline > 0:
                review_questions.append((cntline, elements[0]))
            cntline += 1

    return review_questions


def generate_answer(
    querier: Querier, review_question: Tuple[int, str, str]
) -> Tuple[str, str]:
    """
    Generate an answer to the given question with the provided Querier instance

    Parameters
    ----------
    querier : Querier
        the Querier object
    review_question : Tuple[int, str, str]
        tuple of question id, question type and question

    Returns
    -------
    Tuple[str, str]]
        tuple containing the answer and the associated sources used (in string form)
    """
    # Generate the answer for the question
    if review_question[1].lower() == "initial":
        querier.clear_history()
    response = querier.ask_question(review_question[2])
    source_docs = ""
    for doc in response["source_documents"]:
        source_docs += (
            f"page {str(doc.metadata['page_number'])}\n{doc.page_content}\n\n"
        )

    return response["answer"], source_docs


def write_qa_prompt(myquerier: Querier, output_path: os.PathLike) -> None:
    """
    Writes the question-answer prompt to the output file

    Parameters
    ----------
    output_path : os.PathLike
        path of the output file
    """
    qa_prompt_template = myquerier.get_qa_template(settings.RETRIEVER_PROMPT_TEMPLATE)
    with open(file=output_path, mode="a", encoding="utf8") as file:
        file.write("QA PROMPT TEMPLATE: \n")
        file.write(f"{qa_prompt_template} \n\n")


def write_synthesis_prompt(output_path: os.PathLike) -> None:
    """
    Writes the synthesis prompt to the output file

    Parameters
    ----------
    output_path : os.PathLike
        path of the output file
    """
    with open(file=output_path, mode="a", encoding="utf8") as file:
        file.write("SYNTHESIS PROMPT TEMPLATE: \n")
        file.write(f"{pr.SYNTHESIZE_PROMPT_TEMPLATE} \n\n")


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
        file.write(f"settings.CHUNKS_K_FOR_RERANK =  {settings.CHUNKS_K_FOR_RERANK} \n")
        file.write(f"settings.RETRIEVER_PROMPT_TEMPLATE =  {settings.RETRIEVER_PROMPT_TEMPLATE} \n\n")


def create_answers_for_folder(
    synthesis: str,
    review_files: List[str],
    review_questions: List[Tuple[int, str, str]],
    content_folder_name: str,
    querier: Querier,
    vecdb_folder_path: str,
    output_path: os.PathLike,
) -> None:
    """
    Phase 1 of the review: loop over all the questions and all the documents, gather the answers and store on disk

    Parameters
    ----------
    review_files : List[str]
        list of files to be reviewed
    review_questions : List[Tuple[int, str, str]]
        list of tuples containing question id, question type and question
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
            "sources",
        ]
    )
    cntrow = 0
    for review_file in review_files:
        # create the query chain with a search filter and answer each question for each paper
        querier.make_chain(
            content_folder_name,
            vecdb_folder_path,
            search_filter={"filename": review_file},
        )
        metadata = querier.get_meta_data_by_file_name(review_file)
        for review_question in review_questions:
            logger.info(
                f"reviewing question {review_question[0]} for file: {review_file}"
            )
            cntrow += 1
            # Generate answer
            answer, sources = generate_answer(querier, review_question)
            answer_plus_document_reference = f"This answer is from {metadata['filename']}:\n {answer}"
            final_answer = answer_plus_document_reference if synthesis.lower() == "y" else answer
            df_result.loc[cntrow] = [
                review_file,
                review_question[0],
                review_question[1],
                review_question[2],
                final_answer,
                sources,
            ]

    # First clean up the newlines in the text columns
    def clean_newlines(text):
        if isinstance(text, str):
            return text.replace('\n', ' ')
        return text

    # Apply to columns that contain text with newlines
    text_columns = ['question', 'answer', 'sources']
    for col in text_columns:
        df_result[col] = df_result[col].apply(clean_newlines)

    # Then save to TSV
    df_result = df_result.sort_values(by=["question_id", "filename"])
    df_result.to_csv(output_path,
                     sep='\t',
                     index=False,
                     mode='a',
                     quoting=csv.QUOTE_ALL,
                     encoding='utf-8-sig',
                     escapechar='\\')


def synthesize_results(querier: Querier,
                       results_path: str,
                       output_path: str,
                       synthesis_prompts: List[Tuple[int, str]]) -> None:
    """
    Phase 2 of the review: synthesizes, per question, the results from phase 1

    Parameters

    ----------
    querier : Querier
        the Querier object
    results_path : str
        path of the file resulting from phase 1
    output_path : os.PathLike
        path of the output file
    synthesis_prompts: List[Tuple[int, str]]
        list of tuples containing question id and synthesis prompt
    """
    # load questions and answers
    answers_df = pd.read_csv(filepath_or_buffer=results_path,
                             delimiter="\t")
    # loop over questions
    result = {}
    for question_num in answers_df["question_id"].unique():
        logger.info(f"synthesizing answers for question {question_num}")
        # put all answers to the question in one string
        df_specific_questions = answers_df.loc[
            answers_df["question_id"] == question_num
        ].copy()
        # make sure that the LLM understands a new paper starts
        answer_list = [question for question in df_specific_questions["answer"]]
        answer_string = "\n\n\n\n New Paper:\n".join(answer_list)
        question = df_specific_questions["question"].iloc[0]
        # synthesize results: load prompt for synthesis
        synthesize_prompt_template = synthesis_prompts[question_num-1][1]
        # synthesize_prompt_template = PromptTemplate.from_template(
        #     template=pr.SYNTHESIZE_PROMPT_TEMPLATE
        # )
        synthesis_prompt = str(synthesize_prompt_template).format(
            question=question, answer_string=answer_string
        )
        synthesis = querier.llm.invoke(synthesis_prompt)
        result[question] = synthesis.content
    # write results
    with open(file=output_path, mode="a", newline="", encoding="utf8") as file:
        # create a writer object specifying TAB as delimiter
        tsv_writer = csv.writer(file, delimiter="\t")
        # write the header
        tsv_writer.writerow(["question", "answer"])
        # write data
        for key, value in result.items():
            # remove tabs and new lines
            key = key.replace("\n", " ").replace("\t", " ")
            value = value.replace("\n", " ").replace("\t", " ")
            tsv_writer.writerow([key, value])


def main() -> None:
    """
    Main loop of this module
    """
    # get source folder with papers from user
    content_folder_path = input("Source folder of documents (including path): ")
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    # Get private docs indicator from user
    confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    confidential = confidential_yn in ["y", "Y"]
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
    question_list_path = os.path.join(content_folder_path, "review", "questions.txt")
    synthesis_list_path = os.path.join(content_folder_path, "review", "synthetsize_prompt.txt")

    # if question list path does not exist, stop
    if not os.path.exists(question_list_path):
        logger.info(
            f"The file with questions does not exist, please make sure it exists at {question_list_path}."
        )
        ut.exit_program()

    synthesis = input("Summarize the answers for each question? (y/n): ")

    # get list of relevant files in document folder
    review_files = ut.get_relevant_files_in_folder(content_folder_path)

    # create instance of Querier once
    querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

    # create output folder with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hhour_%Mmin_%Ssec")
    os.mkdir(os.path.join(content_folder_path, f"review/{timestamp}"))
    # copy the questions and synthethis file to the output folder
    os.system(f"cp {question_list_path} {content_folder_path}/review/{timestamp}/questions.txt")
    # ingest documents if documents in source folder path are not ingested yet
    ingest_or_load_documents(content_folder_name=content_folder_name,
                             content_folder_path=content_folder_path,
                             vecdb_folder_path=vecdb_folder_path)

    # get review questions from file
    review_questions = get_review_questions(question_list_path)
    synthesis_prompts = get_synthesis_questions(synthesis_list_path)
    # write out settings
    output_path_settings = os.path.join(
        content_folder_path, f"review/{timestamp}", "settings.txt"
    )
    output_path_synthesis_prompt = os.path.join(
        content_folder_path, f"review/{timestamp}", "synthesis_template.txt"
    )
    output_path_synthesis = os.path.join(
        content_folder_path, f"review/{timestamp}", "synthesis.tsv"
    )
    write_settings(input_path=content_folder_path,
                   confidential=confidential,
                   output_path=output_path_settings)
    # write question-answer template to file
    output_path_qa = os.path.join(
        content_folder_path, f"review/{timestamp}", "qa_template.txt"
    )
    write_qa_prompt(myquerier=querier,
                    output_path=output_path_qa)

    # check if there is already a result, if so skip creation of answers
    output_path_review = os.path.join(
        content_folder_path, f"review/{timestamp}", "answers.tsv"
    )
    create_answers_for_folder(
        synthesis=synthesis,
        review_files=review_files,
        review_questions=review_questions,
        content_folder_name=content_folder_name,
        querier=querier,
        vecdb_folder_path=vecdb_folder_path,
        output_path=output_path_review
    )
    logger.info("Successfully reviewed the documents.")

    if synthesis.lower() == "y":
        write_synthesis_prompt(output_path=output_path_synthesis_prompt)
        # second phase: synthesize the results
        synthesize_results(querier=querier,
                           results_path=output_path_review,
                           output_path=output_path_synthesis,
                           synthesis_prompts=synthesis_prompts)
        logger.info("Successfully synthesized results.")


if __name__ == "__main__":
    main()
