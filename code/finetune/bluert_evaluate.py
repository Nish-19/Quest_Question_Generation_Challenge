import re
import string
import statistics
import evaluate 
import pandas as pd

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


bleurt = evaluate.load('bleurt', 'bleurt-20')

preds_df = pd.read_excel('predictions.xlsx')
val_df = pd.read_csv('../../data/train_val_split_csv/val.csv')

gt_questions = val_df['question'].apply(normalize).tolist()
gen_questions = preds_df['prediction'].apply(normalize).tolist()

all_scores = []
for i in range(0, len(gt_questions), 8):
    if i+8 > len(gt_questions):
        max_limit = len(gt_questions)
    else:
        max_limit = i+8
    gen_batch = gen_questions[i:max_limit]
    gt_batch = gt_questions[i:max_limit]
    scores = bleurt.compute(predictions=gen_batch, references=gt_batch)['scores']
    all_scores.extend(scores)

assert len(all_scores) == len(gt_questions)

score_df = pd.DataFrame()
score_df['scores'] = all_scores
score_df.to_csv('t5_scores.csv', index=False)
print('Mean of BLUERT is:', statistics.mean(all_scores))