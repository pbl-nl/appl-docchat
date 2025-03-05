from pathlib import Path
import pandas as pd
from collections import Counter

results_folder_path = input("Source folder path: ")
# folder_path = Path('X:\\User\\troosts\\projects\\appl-docchat\\docs\\GBF_T7_all_pl\\review\\' + source_folder_path + '\\synthesis.tsv')
folder_path = Path(results_folder_path + '\\synthesis.tsv')
df_syn = pd.read_csv(folder_path, sep='\t')
result = df_syn["answer"].str.strip().values[0]
results = result.split("\n")
clusters = [cluster_result.split(":")[1].strip() for cluster_result in results]
# Count the occurrences of each string
string_counts = Counter(clusters)
print(string_counts)
