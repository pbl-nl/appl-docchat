from pathlib import Path
import pandas as pd
import regex as re
import os
import itertools

def strip_alphanumeric(s):
    # Use a regular expression to remove alphanumeric characters from the start and end of the string
    return re.sub(r'^\D+|\D+$', '', s)

# write string to a .txt file
def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def generate_permutations(input_list):
    permutations = []
    for r in range(1, len(input_list) + 1):
        permutations.extend(itertools.permutations(input_list, r))
    return permutations

def classify_string(s: str) -> list[int]:
    if type(s) != str:
        return []
    if s.isdigit():
        return [int(s)]
    elif ', ' in s:
        classifications = s.split(', ')
    elif ',' in s:
        classifications = s.split(',')
    elif ' ' in s:
        classifications = s.split(' ')
    else:
        return None
    return [int(c) for c in classifications if c.isdigit()]
    
source_folder_path = input("Source folder path: ")
results_folder_path = os.path.join(source_folder_path, "review")

# The selection string is the string that is common for all output folders you want to consider aggregating
# e.g. If you want to obtain the majority vote for resulting classifications in folders:
# 2025_05_13_12hour_02min_48sec
# 2025_05_13_12hour_03min_01sec
# 2025_05_13_12hour_03min_11sec
# 2025_05_20_16hour_18min_39sec
# 2025_05_20_16hour_19min_27sec
# 2025_05_20_16hour_19min_39sec
# 2025_05_20_16hour_19min_49sec
# 2025_05_20_16hour_19min_59sec
# 2025_05_20_16hour_20min_15sec
# 2025_05_20_16hour_20min_31sec
# 2025_05_20_16hour_20min_44sec
# 2025_05_20_16hour_20min_53sec
# 2025_05_20_16hour_21min_04sec
# 2025_05_20_16hour_21min_15sec
# then you would choose as selection string 2025_05_20_16 to just gather all results from folder starting their name with "2025_05_20_16"
selection_string = input("Selection string: ")
# make a list of all relevant folder names in results_folder_path
folder_names = [f for f in os.listdir(results_folder_path) if f.startswith(selection_string)]
# for each folder in folder_names
first = True
to_classify_results = {}
for counter, folder_name in enumerate(folder_names):
    question_ids = []
    questions = []
    to_classifies = []
    classifications = []
    error_comments = []
    questions_folder_path = Path(results_folder_path + "\\" + folder_name + '\\questions.csv')
    df_questions = pd.read_csv(questions_folder_path, sep=',')
    answers_folder_path = Path(results_folder_path + "\\" + folder_name + '\\answers.tsv')
    df_answers = pd.read_csv(answers_folder_path, sep='\t', engine='python')
    for index, row in df_answers.iterrows():
        question_classification = row['question_classification']
        # only consider classification questions
        if question_classification == "y":
            question_id = row['question_id']
            question = row['question']
            to_classify = row['answer'].split(':')[0]
            # only take classification of the first element to classify
            classification_answer, comment = (row['answer'].split(':')[1].strip(), None) if ':' in row['answer'] else ([], f'Error classifications for {to_classify}')
            # if classification contains a space, take only the part before the space and convert to int
            classification = classify_string(classification_answer)
            if classification is not None:
                classification.sort()
                mapping = row['classes'].split('|')
                # map classification number to classification description
                classification = [mapping[int(classification_value) - 1] if int(classification_value) <= len(mapping) else 'Classification Index out of Range'  for classification_value in classification]

            else:
                classification = None
            question_ids.append(question_id)
            questions.append(question)
            to_classifies.append(to_classify)
            classifications.append(classification)
            error_comments.append(comment)

    if first:
        df_results = pd.DataFrame({"question_id": question_ids,
                                   "question": questions,
                                   "to_classify": to_classifies,
                                   "run_" + str(counter + 1): classifications,
                                   'error_comments': error_comments})
        first = False
    else:
        # append the classes series to the dataframe as a new column
        df_results["run_" + str(counter + 1)] = classifications
    
numruns = counter + 1


run_columns = [col for col in df_results.columns if col.startswith('run_')]
df_results["majority vote"] = df_results[run_columns].mode(axis=1)[0]
df_results["confidence score"] = df_results.apply(
    lambda row: round(
        (row[run_columns].apply(lambda x: str(x) == str(row["majority vote"]))).sum() / numruns, 
        2
    ), 
    axis=1
)

# for each row in the dataframe df_results
for index, row in df_results.iterrows():
    question = row['question']
    classification = df_questions.loc[df_questions['Question'] == question, "Classification"].values[0]
    if classification == "y":
        mapping = df_questions.loc[df_questions['Question'] == question, "Classes"].values[0].split('\n')
        for mapped_class in mapping:
            if mapped_class == df_results["majority vote"][index]:
                df_results.at[index, mapped_class] = 1
            else:
                df_results.at[index, mapped_class] = 0


df_results.to_csv(os.path.join(results_folder_path, "multiple_run_answers.csv"), index=False)

# summarize the results from df_results: make a list of countries and their classifications
content = ""
# get a list of unique questiion ids
question_ids = df_results['question_id'].unique()
# for each question id, get the corresponding question and classification
for question_id in question_ids:
    question = df_results.loc[df_results['question_id'] == question_id, 'question'].values[0]
    classification = df_questions.loc[df_questions['Question'] == question, "Classification"].values[0]
    if classification == "y":
        content += f"question: {question}:\n"
        # get all rows for this question_id
        question_rows = df_results[df_results['question_id'] == question_id]

        # get unique classes that actually appear in the data (explode if it is a list instead of a string)
        if question_rows['majority vote'].apply(lambda x: isinstance(x, list)).any():
            question_rows = question_rows.explode('majority vote')
        unique_classes = question_rows['majority vote'].unique()

        # loop through the unique classes
        for mapped_class in unique_classes:
            class_rows = question_rows[question_rows['majority vote'] == mapped_class]
            total_occurrences = len(question_rows)
            count_occurrences = len(class_rows)

            # add the "to_classify" values (comma-separated)
            values = ", ".join(class_rows['to_classify'].astype(str).tolist())

            content += f"{mapped_class}: {values}\n"
            percentage_occurrences = round(count_occurrences / total_occurrences * 100, 1)
            content += f"percentage of occurrences: {percentage_occurrences}%\n"

        content += "\n\n"


# write the content to a file
write_to_file(file_path=os.path.join(results_folder_path, "multiple_run_answers_summary.txt"), content=content)
