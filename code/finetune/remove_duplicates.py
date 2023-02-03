import string
import re
import os
import pandas as pd

test_path = './data/results/testset'
testfile = 'nucleus_reft_flan_t5_large_nodup_selemaugment_0.95_1.20_150.csv'
test_df = pd.read_csv(os.path.join(test_path, testfile))

# BLEURT functions
def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

cleaned_questions = test_df['generated_question'].apply(normalize)
test_df['generated_question'] = cleaned_questions

# TODO: Remove duplicate questions
min_len = 100000 # some high number
pair_id_list, unique_ques_list = [], []
grp_pair_ids = test_df.groupby('pair_id')
for ctr, grp_full in enumerate(grp_pair_ids):
    pair_id, grp = grp_full[0], grp_full[1]
    all_ques = grp['generated_question'].tolist() 
    unique_ques = list(set(all_ques))
    unique_ques_list.extend(unique_ques)
    unique_ques_len = len(unique_ques)
    pair_id_list.extend([pair_id for _ in range(unique_ques_len)])
    if unique_ques_len < min_len:
        min_len = unique_ques_len

no_dup_df = pd.DataFrame()
no_dup_df['pair_id'] = pair_id_list
no_dup_df['generated_question'] = unique_ques_list
filename, ext = os.path.splitext(testfile)
no_dup_df.to_csv(os.path.join(test_path, filename+'_no_duplicate.csv'), index=False)

print('Least number of questions for a given pair id is:', min_len)