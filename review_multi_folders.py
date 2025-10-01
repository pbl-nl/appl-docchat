"""
This module is meant to use the review.py module repetitively
"""
import review
import pandas as pd
from chromadb.api.shared_system_client import SharedSystemClient

if __name__ == "__main__":
    # get source folder with papers from user
    content_folder_paths_filename = \
        input("Enter path and filename of file with folders with questions to run sequentially: ")
    content_folder_paths = pd.read_csv(content_folder_paths_filename, header=None, names=['folder']).apply(
        lambda x: x['folder'], axis=1).tolist()
    print(content_folder_paths)
    # get number of repetitions
    repetitions = input("How many repetitions? : ")
    for content_folder_path in content_folder_paths:
        for i in range(1, int(repetitions) + 1):
            review.main(content_folder_path)
            # clear the cache of chromadb to allow multiple runs sequentially
            SharedSystemClient.clear_system_cache()
