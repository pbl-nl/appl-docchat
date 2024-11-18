import os
from typing import List, Tuple
import json
from collections import defaultdict
import pandas as pd
from loguru import logger
from datasets import Dataset
import langchain.docstore.document as docstore
from ragas import evaluation
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import settings
import utils as ut


def get_eval_questions(evaluation_folder: str, eval_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    reads the json file with evaluation questions

    Parameters
    ----------
    evaluation_folder : str
        name of content folder (without path)
    eval_file : str
        name of the evaluation file

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        tuple of evaluation questions list, evaluation questions types (initial or followup) list and
        ground truths list
    """
    # Get question types, questions and ground_truth from json file
    with open(os.path.join(settings.EVAL_DIR, eval_file), 'r', encoding="utf8") as eval_file:
        evaluation_data = json.load(eval_file)
    eval_question_files = [el["file"] if "file" in el else "" for el in evaluation_data[evaluation_folder]]
    eval_question_types = [el["question_type"] for el in evaluation_data[evaluation_folder]]
    eval_questions = [el["question"] for el in evaluation_data[evaluation_folder]]
    eval_groundtruths = [el["ground_truth"] for el in evaluation_data[evaluation_folder]]

    return eval_questions, eval_question_files, eval_question_types, eval_groundtruths


def ingest_or_load_documents(evaluation_folder: str,
                             content_folder_path: str,
                             vectordb_folder_path: str) -> None:
    """
    ingests documents and creates vector store or just loads documents if vector store already exists

    Parameters
    ----------
    evaluation_folder : str
        name of content folder (without path)
    content_folder_path : str
        name of content folder (including path)
    vectordb_folder_path : str
        name of associated vector store (including path)
    """
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vectordb_folder_path):
        # ingest documents
        ingester = Ingester(collection_name=evaluation_folder,
                            content_folder=content_folder_path,
                            vecdb_folder=vectordb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {evaluation_folder}")


def generate_answer(querier: Querier,
                    eval_question: str,
                    eval_question_type: str) -> Tuple[str, List[docstore.Document]]:
    """
    invokes the chain to generate answers for the questions in the question list

    Parameters
    ----------
    querier : Querier
        Querier object used to ask question
    eval_question : str
        evaluation question
    eval_question_type : str
        evaluation question type

    Returns
    -------
    Tuple[str, List[docstore.Document]]
        tuple of answers to the questions and sources used
    """
    # Iterate over the questions and generate the answers
    logger.info(f"question = {eval_question}")
    if eval_question_type == "initial":
        querier.clear_history()
    response = querier.ask_question(eval_question)

    return response["answer"], response['source_documents']


def get_ragas_results(answers: List[str],
                      sources: List[str],
                      eval_questions: List[str],
                      eval_groundtruths: List[str]) -> evaluation.Result:
    """
    runs the ragas evaluations

    Parameters
    ----------
    answers : List[str]
        list of answers that were generated as a result of the invocation of the chainwas read
    sources : List[str]
        list of sources that was stored as a result of the invocation of the chain
    eval_questions : List[str]
        list of questions from the json file that was read
    eval_groundtruths : List[str]
        list of human expert composed answers as described in the question list json file that was read

    Returns
    -------
    evaluation.Result
        the result that the ragas package produces, in the form of a number of performance metrics per question
    """
    # create list of dictionaries with the examples consisting of questions and ground_truth, answer, source_documents
    examples = [{"query": eval_question, "ground_truth": eval_groundtruths[i]}
                for i, eval_question in enumerate(eval_questions)]
    # create list of dictionaries with the generated answers and source_documents
    results = [{"result": answers[i], "source_documents": sources[i]} for i in range(len(eval_questions))]

    # prepare for ragas evaluation
    dataset_dict = defaultdict(list)
    for i, example in enumerate(examples):
        dataset_dict["question"].append(example["query"])
        dataset_dict["ground_truth"].append(example["ground_truth"])
        dataset_dict["answer"].append(results[i]["result"])
        if len(results[i]["source_documents"]) > 0:
            dataset_dict["contexts"].append([d.page_content for d in results[i]["source_documents"]])
        else:
            dataset_dict["contexts"].append([""])

    dataset = Dataset.from_dict(dataset_dict)

    # evaluate
    result = evaluation.evaluate(dataset)

    return result


def store_aggregated_results(timestamp: str,
                             admin_columns: List[str],
                             evaluation_folder: str,
                             eval_file: str,
                             result: evaluation.Result) -> None:
    """
    writes aggregated ragas results to file, including some admin columns and all the settings
    one line per folder

    Parameters
    ----------
    timestamp : str
        timestamp of generation of the result files
    admin_columns : List[str]
        some identifiers like content folder name, evaluation file name and timestamp
    evaluation_folder : str
        name of content folder (without path)
    eval_file : str
        name of the evaluation file
    result : evaluation.Result
        the resulting ragas performance metrics
    """
    # administrative data
    admin_data = zip([evaluation_folder], [timestamp], [eval_file])
    df_admin = pd.DataFrame(data=list(admin_data), columns=admin_columns)

    # evaluation results
    agg_columns = list(result.keys())
    agg_data = list(result.values())
    df_agg_result = pd.DataFrame(data=[agg_data], columns=agg_columns)

    # No ragas_score available in ragas package version 1.0.9
    df_agg_result = df_agg_result.loc[:, ["answer_relevancy", "context_precision", "faithfulness", "context_recall"]]

    # gather settings
    settings_dict = ut.get_settings_as_dictionary("settings.py")
    settings_columns = list(settings_dict.keys())
    settings_data = [list(settings_dict.values())[i] for i in range(len(list(settings_dict.keys())))]
    df_settings = pd.DataFrame(data=[settings_data], columns=settings_columns)

    # combined
    df_agg = pd.concat([df_admin, df_agg_result, df_settings], axis=1)

    # add result to existing evaluation file (if that exists) and store to disk
    store_evaluation_result(df_agg, evaluation_folder, "aggregated")


def store_detailed_results(timestamp: str,
                           admin_columns: List[str],
                           evaluation_folder: str,
                           eval_file: str,
                           eval_questions: List[str],
                           eval_question_files: List[str],
                           result: evaluation.Result) -> None:
    """
    writes detailed ragas results to file, including some admin columns and all the questions, answers,
    ground truths and sources used. One line per question


    Parameters
    ----------
    timestamp : str
        timestamp of generation of the result files
    admin_columns : List[str]
        some identifiers like content folder name, evaluation file name and timestamp
    evaluation_folder : str
        name of content folder (without path)
    eval_file : str
        name of the evaluation file
    eval_questions : List[str]
        list of questions from the json file that was read
    eval_question_files : List[str]
        list of files corresponding to the list of questions
    result : evaluation.Result
        the resulting ragas performance metrics
    """
    # administrative data
    folder_data = [evaluation_folder for _ in range(len(eval_questions))]
    timestamp_data = [timestamp for _ in range(len(eval_questions))]
    eval_file_data = [eval_file for _ in range(len(eval_questions))]

    admin_data = list(zip(folder_data, timestamp_data, eval_file_data, eval_question_files))
    df_admin = pd.DataFrame(data=admin_data, columns=admin_columns)

    # evaluation results
    df_result = result.to_pandas().loc[:, ["question", "ground_truth", "answer", "contexts", "answer_relevancy",
                                           "context_precision", "faithfulness", "context_recall"]]
    # combined
    df = pd.concat([df_admin, df_result], axis=1)
    # add result to existing evaluation file (if that exists) and store to disk
    store_evaluation_result(df, evaluation_folder, "detail")


def store_evaluation_result(df: pd.DataFrame, evaluation_folder: str, evaluation_type: str) -> None:
    """
    stores the evaluation results in a csv file, either aggregated or detailed depending on evaluation_type argument

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be stored, either aggregated or detailed
    evaluation_folder : str
        name of content folder (without path)
    evaluation_type : str
        indicator whether or not dataframe to store is aggregated or not
    """
    if evaluation_type == "aggregated":
        path = os.path.join(settings.EVAL_DIR, "results", evaluation_folder + "_agg.tsv")
    else:
        path = os.path.join(settings.EVAL_DIR, "results", evaluation_folder + ".tsv")
    if os.path.isfile(path):
        df_old = pd.read_csv(path, sep="\t")
        df = pd.concat([df, df_old], axis=0)
    df.to_csv(path, sep="\t", index=False)


def main(chunk_size: int = None, chunk_overlap: int = None, chunk_k: int = None):
    """
    main evaluation function that ingests and queries documents according to the evaluation json file

    Parameters
    ----------
    chunk_size : int, optional
        the maximum allowed size of the chunks, by default None
    chunk_overlap : int, optional
        the overlap between two succeeding chunks, by default None
    chunk_k : int, optional
        the maximum number of chunks to return from the retriever, by default None
    """
    confidential = False
    # get relevant models
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(confidential)
    # Create instance of Querier
    querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

    # Get evaluation file name
    eval_file = input("Name of evaluation file (without path): ")

    # Get source folder with evaluation documents from user
    with open(os.path.join(settings.EVAL_DIR, eval_file), mode='r', encoding='utf8') as evalfile:
        eval_file_json = json.load(evalfile)

    folder_list = eval_file_json.keys()
    for evaluation_folder in folder_list:
        # get associated source folder path and vectordb path
        content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name=evaluation_folder,
                                                                            chunk_size=chunk_size,
                                                                            chunk_overlap=chunk_overlap)

        # ingest documents if documents in source folder path are not ingested yet
        ingest_or_load_documents(evaluation_folder=evaluation_folder,
                                 content_folder_path=content_folder_path,
                                 vectordb_folder_path=vectordb_folder_path)

        # Get question types, questions and ground_truth from json file
        eval_questions, eval_question_files, eval_question_types, eval_groundtruths = \
            get_eval_questions(evaluation_folder=evaluation_folder,
                               eval_file=eval_file)

        # Iterate over the questions and generate the answers
        answers = []
        sources = []
        for i, eval_question in enumerate(eval_questions):
            # if a specific file is given for the question, define it as a search filter
            search_filter = None
            if eval_question_files[i] != "":
                search_filter = {"filename": eval_question_files[i]}
            # create the query chain, filtering the vector store on filename if applicable
            querier.make_chain(content_folder=evaluation_folder,
                               vecdb_folder=vectordb_folder_path,
                               search_filter=search_filter)
            # obtain an answer to the question
            eval_question_type = eval_question_types[i]
            current_answer, current_sources = generate_answer(querier=querier,
                                                              eval_question=eval_question,
                                                              eval_question_type=eval_question_type)
            answers.append(current_answer)
            sources.append(current_sources)

        # get for ragas evaluation values
        result = get_ragas_results(answers=answers,
                                   sources=sources,
                                   eval_questions=eval_questions,
                                   eval_groundtruths=eval_groundtruths)

        # update location for results
        if chunk_size:
            evaluation_folder = f"{evaluation_folder}_size_{chunk_size}_overlap_{chunk_overlap}_k_{chunk_k}"

        # store aggregate results including the ragas score:
        timestamp = ut.get_timestamp()
        admin_columns_agg = ["folder", "timestamp", "eval_file"]
        store_aggregated_results(timestamp=timestamp,
                                 admin_columns=admin_columns_agg,
                                 evaluation_folder=evaluation_folder,
                                 eval_file=eval_file,
                                 result=result)

        # store detailed results:
        admin_columns_det = ["folder", "timestamp", "eval_file", "file"]
        store_detailed_results(timestamp=timestamp,
                               admin_columns=admin_columns_det,
                               evaluation_folder=evaluation_folder,
                               eval_file=eval_file,
                               eval_questions=eval_questions,
                               eval_question_files=eval_question_files,
                               result=result)


if __name__ == "__main__":
    main()
