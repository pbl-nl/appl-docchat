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
import utils


def get_settings_dictionary(file_name):
    # Initialize an empty dictionary to store the variables and their values
    variables_dict = {}
    # Open and read the file
    with open(file_name, 'r') as file:
        lines = file.readlines()
    start_reading = False
    # Process each line in the file
    for line in lines:
        # start reading below the line with ####
        if line.startswith("####"):
            start_reading = True
        # ignore comment lines
        if start_reading and not line.startswith("#"):
            # Remove leading and trailing whitespace and split the line by '='
            parts = line.strip().split('=')
            # Check if the line can be split into two parts
            if len(parts) == 2:
                # Extract the variable name and value
                variable_name = parts[0].strip()
                variable_value = parts[1].strip()
                # Use exec() to assign the value to the variable name
                exec(f'{variable_name} = {variable_value}')
                # Add the variable and its value to the dictionary
                variables_dict[variable_name] = eval(variable_name)
    return variables_dict

def main():
    # Create instance of Querier
    querier = Querier()

    # only now import ragas evaluation related modules because ragas requires the openai api key to be set on beforehand
    from ragas import evaluation

    # Get source folder with evaluation documents from user
    content_folder_name = input("Source folder of evaluation documents (without path): ")
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
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
    
    # Get questions and ground_truth from json file
    with open(os.path.join(settings.EVAL_DIR, settings.EVAL_FILE_NAME), 'r') as eval_file:
        eval = json.load(eval_file)
    eval_questions = eval[content_folder_name]["question"]
    eval_groundtruths = eval[content_folder_name]["ground_truth"]

    # Iterate over the questions and generate the answers
    answers = []
    sources = []
    for i, question in enumerate(eval_questions):
        logger.info(f"i = {i}, question_type = {eval[content_folder_name]['question_type'][i]}")
        if eval[content_folder_name]["question_type"][i] == "initial":
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

    # store aggregate results including the ragas score
    agg_columns = list(result.keys())
    agg_scores = list(result.values())
    admin_columns = ["folder", "timestamp"]
    folder = content_folder_name
    timestamp = str(dt.datetime.now())
    agg_data = [folder] + [timestamp] + agg_scores
    df_agg = pd.DataFrame(data=[agg_data], columns=admin_columns + agg_columns)
    df_agg.rename(columns={"context_relevancy": "context_precision"}, errors="raise")
    # add result to existing evaluation file (if that exists) and store to disk
    if os.path.isfile(os.path.join(settings.EVAL_DIR, "eval_agg_" + content_folder_name + ".tsv")):
        df_agg_old = pd.read_csv(os.path.join(settings.EVAL_DIR, "eval_agg_" + content_folder_name + ".tsv"), sep="\t")
        df_agg = pd.concat([df_agg_old, df_agg], axis=0)
    df_agg.to_csv(os.path.join(settings.EVAL_DIR, "eval_agg_" + content_folder_name + ".tsv"), sep="\t", index=False)

    # store detailed results
    settings_dict = get_settings_dictionary("settings.py")
    settings_columns = list(settings_dict.keys())
    combined_columns = admin_columns + settings_columns
    # gather all data
    folder_data = [folder for _ in range(len(eval_questions))]
    timestamp_data = [timestamp for _ in range(len(eval_questions))]
    settings_data = [[list(settings_dict.values())[i] for _ in range(len(eval_questions))] for i in range(len(list(settings_dict.keys())))]
    combined_data = zip(folder_data, timestamp_data, *settings_data)
    df_combined = pd.DataFrame(list(combined_data), columns=combined_columns)
    df_eval = result.to_pandas()
    df = pd.concat([df_combined, df_eval], axis=1)
    # line below is necessary as long as ragas package doesn't update metric name "context_relevancy" to "context_precision"
    df.rename(columns={"context_relevancy": "context_precision"}, errors="raise")
    # add result to existing evaluation file (if that exists) and store to disk
    if os.path.isfile(os.path.join(settings.EVAL_DIR, "eval_" + content_folder_name + ".tsv")):
        df_old = pd.read_csv(os.path.join(settings.EVAL_DIR, "eval_" + content_folder_name + ".tsv"), sep="\t")
        df = pd.concat([df_old, df], axis=0)
    df.to_csv(os.path.join(settings.EVAL_DIR, "eval_" + content_folder_name + ".tsv"), sep="\t", index=False)

    
if __name__ == "__main__":
    main()

    
