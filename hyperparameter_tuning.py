"""
Grid search to tune chunking hyperparameters
"""
import evaluate

# define grid of hyperparameters
CHUNK_SIZE_SET = [250, 1000]
CHUNK_OVERLAP_SET = [200]
CHUNK_K_SET = [2, 4]

# loop over hyperparameters using grid search
for chunk_size in CHUNK_SIZE_SET:
    for chunk_overlap in CHUNK_OVERLAP_SET:
        for chunk_k in CHUNK_K_SET:
            # print current set of hyperparameters
            print(f"chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, chunk_k: {chunk_k}")

            # evaluate performance of hyperparameter set on all benchmark datasets
            evaluate.main(chunk_size, chunk_overlap, chunk_k)
