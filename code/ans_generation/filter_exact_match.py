import os
import string
import pandas as pd

file_dir = './data/results/data_augmentation'
filename = 'answer_all_augmented_with_balanced_question_attribute_parsed_greedy.csv'
file_path = os.path.join(file_dir, filename)
df = pd.read_csv(file_path)

# TODO: clean string
def clean_str(ans):
    return ans.translate(str.maketrans('', '', string.punctuation)).lower()

valid_rows = []
for i, row in df.iterrows():
    if clean_str(row['answer']) == clean_str(row['generated_answer']):
        valid_rows.append(row.values.tolist())

# Construct the dataframe
em_df = pd.DataFrame(valid_rows, columns=df.columns)
filen, filext = os.path.splitext(filename)
em_filen = filen + '_em.csv'
em_file_path = os.path.join(file_dir, em_filen)
em_df.to_csv(em_file_path, index=False)