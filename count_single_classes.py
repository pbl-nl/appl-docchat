from pathlib import Path
import pandas as pd
import regex as re
import os

def strip_alphanumeric(s):
    # Use a regular expression to remove alphanumeric characters from the start and end of the string
    return re.sub(r'^\D+|\D+$', '', s)

# write string to a .txt file
def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

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
multilabel = input("More than 1 class possible? (y/n): ")
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
    questions_folder_path = Path(results_folder_path + "\\" + folder_name + '\\questions.csv')
    df_questions = pd.read_csv(questions_folder_path, sep=',')
    answers_folder_path = Path(results_folder_path + "\\" + folder_name + '\\answers.tsv')
    df_answers = pd.read_csv(answers_folder_path, sep='\t')
    for index, row in df_answers.iterrows():
        question_classification = row['question_classification']
        # only consider classification questions
        if question_classification == "y":
            question_id = row['question_id']
            question = row['question']
            to_classify = row['answer'].split(':')[0]
            # only take classification of the first element to classify
            classification = row['answer'].split(':')[1].strip()
            classification = int(classification.split(' ')[0]) if ' ' in classification else int(classification)
            mapping = row['classes'].split('\n')
            if len(mapping) > 0:
                # map classification number to classification description
                classification = mapping[int(classification) - 1]
            question_ids.append(question_id)
            questions.append(question)
            to_classifies.append(to_classify)
            classifications.append(classification)

    if first:
        df_results = pd.DataFrame({"question_id": question_ids,
                                   "question": questions,
                                   "to_classify": to_classifies,
                                   "run_" + str(counter + 1): classifications})
        first = False
    else:
        # append the classes series to the dataframe as a new column
        df_results["run_" + str(counter + 1)] = classifications

    if multilabel == "y":
        # convert the column to a list of numbers
        df_results["run_" + str(counter + 1)] = \
            df_results["run_" + str(counter + 1)].apply(lambda x: [str(i).strip() for i in x.split(',')])

numruns = counter + 1

print(df_results)
print()
run_columns = [col for col in df_results.columns if col.startswith('run_')]
df_results["majority vote"] = df_results[run_columns].mode(axis=1)[0]
df_results["confidence score"] = \
    df_results.apply(lambda row: round((row[run_columns] == row["majority vote"]).sum() / numruns, 2), axis=1)

print(df_questions)
# for each row in the dataframe df_results
for index, row in df_results.iterrows():
    question = row['question']
    classification = df_questions.loc[df_questions['Question'] == question, "classification"].values[0]
    # print(classification)
    if classification == "y":
        mapping = df_questions.loc[df_questions['Question'] == question, "classes"].values[0].split('\n')
        # print(mapping)
        # mapping = row['classes'].split('\n')
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
    classification = df_questions.loc[df_questions['Question'] == question, "classification"].values[0]
    if classification == "y":
        content += f"question: {question}:\n"
        # get the corresponding classes
        mapping = df_questions.loc[df_questions['Question'] == question, "classes"].values[0].split('\n')
        # for each class, get the value from column "to_classify" for which the value in column "majority vote" is equal to the class
        for mapped_class in mapping:
            total_occurrences = 0
            count_occurrences = 0
            first_occurrence = True
            content += f"{mapped_class}: "
            for index, row in df_results.iterrows():
                if row['question_id'] == question_id:
                    total_occurrences +=1
                    if row['majority vote'] == mapped_class:
                        if first_occurrence:
                            content += f"{row['to_classify']}"
                            first_occurrence = False
                        else:
                            content += f", {row['to_classify']}"
                        count_occurrences += 1
            content += "\n"
            percentage_occurences = round(count_occurrences / total_occurrences * 100, 1)
            content += f"percentage of occurrences: {percentage_occurences}%\n"
        content += "\n\n"


# write the content to a file
write_to_file(file_path=os.path.join(results_folder_path, "multiple_run_answers_summary.txt"), content=content)
