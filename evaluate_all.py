import os
import json
import pandas as pd
# local imports
from query.querier import Querier
import settings
import utils as ut
import evaluate as ev


def main(chunk_size=None, chunk_overlap=None, chunk_k=None):
    # Create instance of Querier
    querier = Querier(chunk_k=chunk_k)

    # Get source folder with evaluation documents from user
    with open(os.path.join(settings.EVAL_DIR, settings.EVAL_FILE_NAME), 'r') as eval_file:
        eval = json.load(eval_file)
    folder_list = eval.keys()
    for folder in folder_list:
        content_folder_name = folder
        # get associated source folder path and vectordb path
        content_folder_path, vectordb_folder_path = ut.create_vectordb_name(content_folder_name, chunk_size, chunk_overlap)
        
        # ingest documents if documents in source folder path are not ingested yet
        ev.ingest_or_load_documents(content_folder_name, content_folder_path, vectordb_folder_path)
        
        # create the query chain
        querier.make_chain(content_folder_name, vectordb_folder_path)
        
        # Get question types, questions and ground_truth from json file
        eval_questions, eval_question_types, eval_groundtruths = ev.get_eval_questions(content_folder_name)
    
        # Iterate over the questions and generate the answers
        answers, sources = ev.generate_answers(querier, eval_questions, eval_question_types)
    
        # get for ragas evaluation values
        result = ev.get_ragas_results(answers, sources, eval_questions, eval_groundtruths)

        #update location for results
        if chunk_size:
            content_folder_name = "{}_size_{}_overlap_{}_k_{}".format(folder, chunk_size, chunk_overlap, chunk_k)

        # store aggregate results including the ragas score:
        timestamp = ut.get_timestamp()
        admin_columns = ["folder", "timestamp"]
        ev.store_aggregated_results(timestamp, admin_columns, content_folder_name, result)
    
        # store detailed results:
        ev.store_detailed_results(timestamp, admin_columns, content_folder_name, eval_questions, result)


if __name__ == "__main__":
    main()

    
