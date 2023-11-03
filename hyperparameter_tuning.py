"""Grid search to tune chunking hyperparameters"""

#load libraries
import evaluate_all

#define grid of hyperparameters
CHUNK_SIZE_SET    = [250, 1000]
CHUNK_OVERLAP_SET = [200]
CHUNK_K_SET       = [2, 4]

#loop over hyperparameters using grid search
for chunk_size in CHUNK_SIZE_SET:
    for chunk_overlap in CHUNK_OVERLAP_SET:
        for chunk_k in CHUNK_K_SET:
            #print current set of hyperparameters
            print("chunk_size: {}, chunk_overlap: {}, chunk_k: {}".format(
                chunk_size, chunk_overlap, chunk_k))

            #evaluate performance of hyperparameter set on all benchmark datasets
            evaluate_all.main(chunk_size, chunk_overlap, chunk_k)

