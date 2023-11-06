import os
from loguru import logger
import json
import pandas as pd
import datetime as dt
from collections import defaultdict
from datasets import Dataset
# local imports
from ingest.ingester import Ingester
from query.querier import Querier
import settings
import utils as ut


def store_evaluation_result(df, content_folder_name, type):
    if type == "aggregated":
        path = os.path.join(settings.EVAL_DIR, content_folder_name + "_agg.tsv")
    else:
        path = os.path.join(settings.EVAL_DIR, content_folder_name + ".tsv")
    if os.path.isfile(path):
        df_old = pd.read_csv(path, sep="\t")
        df = pd.concat([df, df_old], axis=0)
    df.to_csv(path, sep="\t", index=False)


def main():
    # Create instance of Querier
    querier = Querier()

    # only now import ragas evaluation related modules because ragas requires the openai api key to be set on beforehand
    from ragas import evaluation

    # Get source folder with evaluation documents from user
    content_folder_name = input("Source folder of evaluation documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vectordb_folder_path):
        # ingest documents
        ingester = Ingester(content_folder_name, content_folder_path, vectordb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vectordb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")
    
    # create the query chain
    querier.make_chain(content_folder_name, vectordb_folder_path)
    
    # Get question types, questions and ground_truth from json file
    with open(os.path.join(settings.EVAL_DIR, settings.EVAL_FILE_NAME), 'r') as eval_file:
        eval = json.load(eval_file)
    eval_question_types = [el["question_type"] for el in eval[content_folder_name]]
    eval_questions = [el["question"] for el in eval[content_folder_name]]
    eval_groundtruths = [el["ground_truth"] for el in eval[content_folder_name]]

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

    # create list of dictionaries with the examples consisting of questions and ground_truths, answer, source_documents
    examples = [{"query": eval_question, "ground_truths": [eval_groundtruths[i]]} for i, eval_question in enumerate(eval_questions)] 
    # create list of dictionaries with the generated answers and source_documents
    results = [{"result": answers[i], "source_documents": sources[i]} for i in range(len(eval_questions))]

    # prepare for ragas evaluation 
    dataset_dict = defaultdict(list)
    for i, example in enumerate(examples):
        dataset_dict["question"].append(example["query"])
        dataset_dict["ground_truths"].append(example["ground_truths"])
        dataset_dict["answer"].append(results[i]["result"])
        dataset_dict["contexts"].append([d.page_content for d in results[i]["source_documents"]])
    dataset = Dataset.from_dict(dataset_dict)
    # evaluate
    result = evaluation.evaluate(dataset)

    # store aggregate results including the ragas score:
    agg_columns = list(result.keys())
    agg_scores = list(result.values())
    admin_columns = ["folder", "timestamp"]
    timestamp = str(dt.datetime.now())
    agg_data = [content_folder_name] + [timestamp] + agg_scores
    df_agg = pd.DataFrame(data=[agg_data], columns=admin_columns + agg_columns)
    df_agg = df_agg.loc[:, ["folder", "timestamp", "ragas_score", "answer_relevancy", "context_relevancy", "faithfulness", "context_recall"]]
    # line below is necessary as long as ragas package doesn't update metric name "context_relevancy" to "context_precision"
    df_agg = df_agg.rename(columns={"context_relevancy": "context_precision"}, errors="raise")
    # add settings
    settings_dict = ut.get_settings_as_dictionary("settings.py")
    settings_columns = list(settings_dict.keys())
    settings_data = [list(settings_dict.values())[i] for i in range(len(list(settings_dict.keys())))]
    df_settings = pd.DataFrame(data=[settings_data], columns=settings_columns)
    # combined
    df_agg = pd.concat([df_agg, df_settings], axis=1)
    # add result to existing evaluation file (if that exists) and store to disk
    store_evaluation_result(df_agg, content_folder_name, "aggregated")

    # store detailed results:
    # administrative data
    folder_data = [content_folder_name for _ in range(len(eval_questions))]
    timestamp_data = [timestamp for _ in range(len(eval_questions))]
    admin_data = zip(folder_data, timestamp_data)
    df_admin = pd.DataFrame(data=list(admin_data), columns=admin_columns)
    # evaluation results
    df_result = result.to_pandas().loc[:,["question", "answer", "answer_relevancy", "context_relevancy", "faithfulness", "context_recall", 
                                          "contexts", "ground_truths"]]
    # line below is necessary as long as ragas package doesn't update metric name "context_relevancy" to "context_precision"
    df_result = df_result.rename(columns={"context_relevancy": "context_precision"}, errors="raise")
    # combined
    df = pd.concat([df_admin, df_result], axis=1)
    # add result to existing evaluation file (if that exists) and store to disk
    store_evaluation_result(df, content_folder_name, "detail")

    
if __name__ == "__main__":
    main()

    
