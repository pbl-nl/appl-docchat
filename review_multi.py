"""
This module is meant to use the review.py module repetitively
"""
import review
from chromadb.api.shared_system_client import SharedSystemClient

if __name__ == "__main__":
    # get source folder with papers from user
    content_folder_path = input("Source folder of documents (including path): ")
    # get number of repetitions
    repetitions = input("How many repetitions? : ")
    for i in range(1, int(repetitions) + 1):
        review.main(content_folder_path)
        # clear the cache of chromadb to allow multiple runs sequentially
        SharedSystemClient.clear_system_cache()
