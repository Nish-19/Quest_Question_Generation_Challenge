import string
import re
import os
import pandas as pd

test_path = './data/results/data_augmentation'
testfile = 'answer_combine_val_pred_for_em_greedy.csv'
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

# TODO: Drop rows based on duplicate generated_question 
no_dup_df = test_df.drop_duplicates(subset='generated_question', keep="first")


filename, ext = os.path.splitext(testfile)
no_dup_df.to_csv(os.path.join(test_path, filename+'_no_duplicate.csv'), index=False)