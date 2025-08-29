"""
This module is meant to use the review.py module repetitively
"""
import review
import time

if __name__ == "__main__":
    # get source folder with papers from user
    content_folder_path = input("Source folder of documents (including path): ")
    # get number of repetitions
    repetitions = input("How many repetitions? : ")
    for i in range(1, int(repetitions) + 1):
        review.main(content_folder_path)
        time.sleep(15)
