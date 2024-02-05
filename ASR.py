import os
from loguru import logger
# local imports
from asr.asr import AutomatedSystematicReview as asr
import utils as ut


def main():
    # Create instance of Querier once
    # Get source folder with docs from user
    content_folder_name = input("Source folder of documents (without path): ")
    question_list_name = input('Please insert file name of question list: ') 
    question_list_name = 'asr/prompt_instructions/' + question_list_name + '.txt'
    # get associated vectordb path
    content_folder, vectordb_folder_path = ut.create_vectordb_name(content_folder_name)

    # If vector store folder does not exist, stop
    if not os.path.exists(content_folder):
        logger.info("This content folder does not exist. Please make sure the spelling is correct")
        ut.exit_program()
    elif not os.path.exists(question_list_name):
        logger.info("This question list does not exist, please make sure this list exists.")
        ut.exit_program()
    else:
        asr_instace = asr(content_folder, question_list_name)
        asr_instace.conduct_review()



if __name__ == "__main__":
    main()
