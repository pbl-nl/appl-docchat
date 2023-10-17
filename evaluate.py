import os
from loguru import logger
import json
import pandas as pd
import datetime as dt
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
    from ragas.langchain.evalchain import RagasEvaluatorChain
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

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

    # define evaluation chains
    eval_chains = {chain.name: RagasEvaluatorChain(metric=chain) for chain in [faithfulness, answer_relevancy, context_precision, context_recall]}
    # get evaluation metrics for all generated context and final answers
    chains_scores = []
    for name, eval_chain in eval_chains.items():
        chain_score_name = f"{name}_score"
        chain_scores = eval_chain.evaluate(examples, results)
        chain_scores_list = [metric_score[chain_score_name] for metric_score in chain_scores]
        chains_scores.append(chain_scores_list)

    # prepare to store the evaluation results
    settings_dict = get_settings_dictionary("settings.py")
    admin_columns = ["folder", "timestamp"]
    settings_columns = list(settings_dict.keys())
    qa_columns = ["question", "ground_truth", "answer", "context"]
    metric_columns = list(eval_chains.keys())
    all_columns = admin_columns + settings_columns + qa_columns + metric_columns
    
    # gather all data
    folder_data = ["verhuismotieven" for _ in range(len(eval_questions))]
    timestamp_data = [dt.datetime.now() for _ in range(len(eval_questions))]
    settings_data = [[list(settings_dict.values())[i] for _ in range(len(eval_questions))] for i in range(len(list(settings_dict.keys())))]
    all_data = zip(folder_data, timestamp_data, *settings_data, eval_questions, eval_groundtruths, answers, sources, *chains_scores)
    df = pd.DataFrame(list(all_data), columns=all_columns)
    logger.info(df[["folder", "question"] + metric_columns])
    # lines below are necessary as long as ragas package doesn't update metric name "context_relevancy" to "context_precision"
    df.rename(columns={"context_relevancy": "context_precision"}, errors="raise")
    # add result to existing evaluation file (if that exists) and store to disk
    if os.path.isfile(os.path.join(settings.EVAL_DIR, "eval.csv")):
        df_old = pd.read_csv(os.path.join(settings.EVAL_DIR, "eval.csv"), sep="\t")
        df = pd.concat([df_old, df], axis=0)
    df.to_csv(os.path.join(settings.EVAL_DIR, "eval.csv"), sep="\t", index=False)


if __name__ == "__main__":
    main()

    
