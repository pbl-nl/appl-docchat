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
    if s.isdigit():
        return [int(s)]
    elif ', ' in s:
        classifications = s.split(', ')
    elif ',' in s:
        classifications = s.split(',')
    elif ' ' in s:
        classifications = s.split(' ')
    else:
        return []
    return [int(c) for c in classifications if c.isdigit()]

def list_txt_files(directory):
    txt_files = []
    # Walk through the directory
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        # Check if it's a file and has a .txt extension
        if os.path.isfile(full_path) and filename.endswith('.txt'):
            # Remove the extension and add to the list
            txt_files.append(os.path.splitext(filename)[0])

    return txt_files

def calculate_class_scores(row, run_columns, numruns):
    """ Calculate class scores for each row based on occurrences in run columns """
    # Get all possible classes from the 'classes' column
    classes_text = row['classes']
    all_classes = [cls.strip() for cls in classes_text.split('|')]
    # Count occurrences of each class across all run columns
    class_counts = {}
    for run_col in run_columns:
        run_value = row[run_col]
        try:
            # Parse the list string (e.g., "['all areas', 'high importance to biodiversity is emphasized']")
            if isinstance(run_value, list):
                for class_name in run_value:
                    if class_name in all_classes:
                        if class_name in class_counts:
                            # If the key exists, increase its value by 1
                            class_counts[class_name] += 1
                        else:
                            # If the key doesn't exist, add it with value 1
                            class_counts[class_name] = 1
        except (ValueError, SyntaxError):
            # If parsing fails, skip this entry
            continue
    # Create tuples with class name and score
    class_scores = []
    relevant_classes = []
    for class_name in all_classes:
        count = class_counts.get(class_name, 0)
        score = count / numruns
        class_scores.append((class_name, score))
        if score > 0.5:
            relevant_classes.append(class_name)
    
    return class_scores, relevant_classes


    
source_folder_path = input("Source folder path: ")
results_folder_path = os.path.join(source_folder_path, "review")

# The selection string is the string that is common for all output folders you want to consider aggregating
# e.g. If you want to obtain the majority vote for resulting classifications in folders:
# 2025_05_13_12hour_02min_48sec
# 2025_05_13_12hour_03min_01sec
# 2025_05_13_12hour_03min_11sec
# then you would choose as selection string 2025_05_20_16 to just gather all results from folder starting their name with "2025_05_20_16"

# NB: this script assumes that answers.tsv files are sorted in order of filename ascending, then question_id ascending!!
selection_string = input("Selection string: ")
# make a list of all relevant folder names in results_folder_path
folder_names = [f for f in os.listdir(results_folder_path) if f.startswith(selection_string)]
# for each folder in folder_names
first = True
to_classify_results = {}
valid_to_classifies = list_txt_files(source_folder_path)

# loop over folder names
for counter, folder_name in enumerate(folder_names):
    question_ids = []
    questions = []
    question_templates = []
    to_classifies = []
    classes = []
    classifications = []
    error_comments = []
    answers_folder_path = os.path.join(results_folder_path, folder_name, 'answers.tsv')
    df_answers = pd.read_csv(answers_folder_path, sep='\t', engine='python')
    # loop over rows in aswers.tsv
    for index, row in df_answers.iterrows():
        question_classification = row['question_classification']
        # only consider classification questions
        if question_classification == "y":
            question_template = row['question_template']
            question_id = row['question_id']
            question = row['question']
            classes_string = row['classes']
            to_classify = row['answer'].split("|")[0].split(':')[0]
            # only take classification of the first element to classify
            classification_answer, comment = (row['answer'].split("|")[0].split(':')[1].strip(), None) if ':' in row['answer'] else ("", f'Error classifications for {to_classify}')
            # if classification contains a space, take only the part before the space and convert to int
            classification = classify_string(classification_answer)
            if classification is not None:
                classification.sort()
                mapping = row['classes'].split('|')
                # map classification number to classification description
                classification = [mapping[int(value) - 1] if int(value) <= len(mapping) else 'Classification Index out of Range' for value in classification]
            else:
                classification = None
            # append lists for insertion into dataframe
            question_ids.append(question_id)
            questions.append(question)
            question_templates.append(question_template)
            to_classifies.append(to_classify)
            classes.append(classes_string)
            classifications.append(classification)
            error_comments.append(comment)

    if first:
        df_results = pd.DataFrame({"question_id": question_ids,
                                   "question": questions,
                                   "question_template": question_templates,
                                   "to_classify": to_classifies,
                                   "classes": classes,
                                   "run_" + str(counter + 1): classifications,
                                   "errors_run_" + str(counter + 1): error_comments})
        first = False
    else:
        # append the classifications to the dataframe as a new column
        df_results["run_" + str(counter + 1)] = classifications
        df_results["errors_run_" + str(counter + 1)] = error_comments

# determine the number of occurences per class
numruns = len(folder_names)
run_columns = [col for col in df_results.columns if col.startswith('run_')]
# Add scores per class and relevant classes to the results dataframe
class_scores_list = []
relevant_classes_list = []
# df_results["class_scores"] = [[] for _ in range(len(df_results))]
for index, row in df_results.iterrows():
    # Apply the function to create the new column
    class_scores, relevant_classes = calculate_class_scores(row, run_columns, numruns)
    class_scores_list.append(class_scores)
    relevant_classes_list.append(relevant_classes)
df_results["class_scores"] = class_scores_list
df_results["relevant_classes"] = relevant_classes_list

# write the content to file multiple_run_answers.csv
df_results.to_csv(os.path.join(results_folder_path, "multiple_run_answers.csv"), index=False)


#### LIST ALL COUNTRIES PER QUESTION AND CLASS ####
# Take file "questions.csv" from first folder in list of folders
questions_folder_path = os.path.join(results_folder_path, folder_names[0], 'questions.csv')
df_questions = pd.read_csv(questions_folder_path, sep=',')
content = ""
# loop over all questions
for _, row in df_questions.iterrows():
    question = row["Question"]
    content += f"question: {question}:\n"
    classes = row['Classes'].split('|')
    for my_class in classes:
        cnt = 0
        content += f"{my_class}: "
        for _, row in df_results.iterrows():
            if my_class in row["relevant_classes"]:
                content += row["to_classify"] + ", "
                cnt += 1
        content = content[:-2] + "\n"
        percentage_occurrences = round(cnt / len(valid_to_classifies) * 100, 1)
        content += f"percentage of occurrences: {percentage_occurrences}%\n"
    content += "\n\n"

# write the content to file multiple_run_answers_summary.txt
write_to_file(file_path=os.path.join(results_folder_path, "multiple_run_answers_summary.txt"), content=content)
