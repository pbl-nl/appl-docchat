import os
# local imports
from query.querier import Querier
from settings import VECDB_DIR, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_TYPE


def main():
    proceed = True
    # Get source folder with docs from user
    input_folder = input("Source folder of documents (without path): ")
    vectordb_name = "_" + VECDB_TYPE + "_" + str(CHUNK_SIZE) + "_" + str(CHUNK_OVERLAP) + "_" + EMBEDDINGS_TYPE
    vectordb_folder = os.path.join(VECDB_DIR, input_folder) + vectordb_name 
    # If vector store folder does not exist, stop
    if not os.path.exists(vectordb_folder):
        print("There is no vector database for this folder yet. First run \"python ingest.py\"")
        proceed = False
    else:
        # Create instance of Querier once
        querier = Querier(input_folder, vectordb_folder, EMBEDDINGS_TYPE, VECDB_TYPE, CHUNK_SIZE, CHUNK_OVERLAP)
        while proceed:
            print()
            # Get question from user
            question = input("Question: ")
            if question != "exit":
                # Generate answer and include sources used to produce that answer
                answer, source = querier.ask_question(question)

                print(f"\nAnswer: {answer}")
                print("\nSources:\n")
                for document in source:
                    print(f"Page {document.metadata['page_number']} chunk used: {document.page_content}\n")
            else:
                proceed = False


if __name__ == "__main__":
    main()
