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


def get_eval_questions(content_folder_name: str, eval_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    reads the json file with evaluation questions

    Parameters
    ----------
    content_folder_name : str
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
    eval_question_types = [el["question_type"] for el in evaluation_data[content_folder_name]]
    eval_questions = [el["question"] for el in evaluation_data[content_folder_name]]
    eval_groundtruths = [el["ground_truth"] for el in evaluation_data[content_folder_name]]

    return eval_questions, eval_question_types, eval_groundtruths


def ingest_or_load_documents(content_folder_name: str,
                             content_folder_path: str,
                             vectordb_folder_path: str) -> None:
    """
    ingests documents and creates vector store or just loads documents if vector store already exists

    Parameters
    ----------
    content_folder_name : str
        name of content folder (without path)
    content_folder_path : str
        name of content folder (including path)
    vectordb_folder_path : str
        name of associated vector store (including path)
    """
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vectordb_folder_path):
        # ingest documents
        ingester = Ingester(collection_name=content_folder_name,
                            content_folder=content_folder_path,
                            vecdb_folder=vectordb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")


def generate_answers(querier: Querier,
                     eval_questions: List[str],
                     eval_question_types: List[str]) -> Tuple[List[str], List[List[docstore.Document]]]:
    """
    invokes the chain to generate answers for the questions in the question list

    Parameters
    ----------
    querier : Querier
        Querier object used to ask questions
    eval_questions : List[str]
        list of evaluation questions
    eval_question_types : List[str]
        list of evaluation questions types

    Returns
    -------
    Tuple[List[str], List[List[docstore.Document]]]
        tuple of answers to the questions and sources used
    """
    # Iterate over the questions and generate the answers
    answers = []
    sources = []
    for i, question in enumerate(eval_questions):
        logger.info(f"i = {i}, question_type = {eval_question_types[i]}")
        if eval_question_types[i] == "initial":
            querier.clear_history()
        response = querier.ask_question(question)
        answers.append(response["answer"])
        sources.append(response["source_documents"])

    return answers, sources


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
        dataset_dict["contexts"].append([d.page_content for d in results[i]["source_documents"]])
    dataset = Dataset.from_dict(dataset_dict)

    # evaluate
    result = evaluation.evaluate(dataset)

    return result


def store_aggregated_results(timestamp: str,
                             admin_columns: List[str],
                             content_folder_name: str,
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
    content_folder_name : str
        name of content folder (without path)
    eval_file : str
        name of the evaluation file
    result : evaluation.Result
        the resulting ragas performance metrics
    """
    # administrative data
    admin_data = zip([content_folder_name], [timestamp], [eval_file])
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
    store_evaluation_result(df_agg, content_folder_name, "aggregated")


def store_detailed_results(timestamp: str,
                           admin_columns: List[str],
                           content_folder_name: str,
                           eval_file: str,
                           eval_questions: List[str],
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
    content_folder_name : str
        name of content folder (without path)
    eval_file : str
        name of the evaluation file
    eval_questions : List[str]
        list of questions from the json file that was read
    result : evaluation.Result
        the resulting ragas performance metrics
    """
    # administrative data
    folder_data = [content_folder_name for _ in range(len(eval_questions))]
    eval_file_data = [eval_file for _ in range(len(eval_questions))]
    timestamp_data = [timestamp for _ in range(len(eval_questions))]
    admin_data = zip(folder_data, timestamp_data, eval_file_data)
    df_admin = pd.DataFrame(data=list(admin_data), columns=admin_columns)

    # evaluation results
    df_result = result.to_pandas().loc[:, ["question", "ground_truth", "answer", "contexts", "answer_relevancy",
                                           "context_precision", "faithfulness", "context_recall"]]

    # combined
    df = pd.concat([df_admin, df_result], axis=1)

    # add result to existing evaluation file (if that exists) and store to disk
    store_evaluation_result(df, content_folder_name, "detail")


def store_evaluation_result(df: pd.DataFrame, content_folder_name: str, evaluation_type: str) -> None:
    """
    stores the evaluation results in a csv file, either aggregated or detailed depending on evaluation_type argument

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be stored, either aggregated or detailed
    content_folder_name : str
        name of content folder (without path)
    evaluation_type : str
        indicator whether or not dataframe to store is aggregated or not
    """
    if evaluation_type == "aggregated":
        path = os.path.join(settings.EVAL_DIR, content_folder_name + "_agg.tsv")
    else:
        path = os.path.join(settings.EVAL_DIR, content_folder_name + ".tsv")
    if os.path.isfile(path):
        df_old = pd.read_csv(path, sep="\t")
        df = pd.concat([df, df_old], axis=0)
    df.to_csv(path, sep="\t", index=False)


def main(chunk_size=None, chunk_overlap=None, chunk_k=None):
    # Create instance of Querier
    querier = Querier(chunk_k=chunk_k)

    # Get evaluation file name
    eval_file = input("Name of evaluation file (without path): ")

    # Get source folder with evaluation documents from user
    with open(os.path.join(settings.EVAL_DIR, eval_file), mode='r', encoding='utf8') as eval:
        eval_file_json = json.load(eval)
    folder_list = eval_file_json.keys()
    for folder in folder_list:
        content_folder_name = folder
        # get associated source folder path and vectordb path
        content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name=content_folder_name,
                                                                            chunk_size=chunk_size,
                                                                            chunk_overlap=chunk_overlap)

        # ingest documents if documents in source folder path are not ingested yet
        ingest_or_load_documents(content_folder_name=content_folder_name,
                                 content_folder_path=content_folder_path,
                                 vectordb_folder_path=vectordb_folder_path)

        # create the query chain
        querier.make_chain(content_folder=content_folder_name,
                           vecdb_folder=vectordb_folder_path)

        # Get question types, questions and ground_truth from json file
        eval_questions, eval_question_types, eval_groundtruths = \
            get_eval_questions(content_folder_name=content_folder_name,
                               eval_file=eval_file)

        # Iterate over the questions and generate the answers
        answers, sources = generate_answers(querier=querier,
                                            eval_questions=eval_questions,
                                            eval_question_types=eval_question_types)

        # get for ragas evaluation values
        result = get_ragas_results(answers=answers,
                                   sources=sources,
                                   eval_questions=eval_questions,
                                   eval_groundtruths=eval_groundtruths)

        # update location for results
        if chunk_size:
            content_folder_name = f"{folder}_size_{chunk_size}_overlap_{chunk_overlap}_k_{chunk_k}"

        # store aggregate results including the ragas score:
        timestamp = ut.get_timestamp()
        admin_columns = ["folder", "timestamp", "eval_file"]
        store_aggregated_results(timestamp=timestamp,
                                 admin_columns=admin_columns,
                                 content_folder_name=content_folder_name,
                                 eval_file=eval_file,
                                 result=result)

        # store detailed results:
        store_detailed_results(timestamp=timestamp,
                               admin_columns=admin_columns,
                               content_folder_name=content_folder_name,
                               eval_file=eval_file,
                               eval_questions=eval_questions,
                               result=result)


if __name__ == "__main__":
    main()
