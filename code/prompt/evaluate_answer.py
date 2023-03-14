import os 
import pandas as pd 
import string 
import re 
from rouge_score import rouge_scorer

def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_answer_tag(text):
        return text.replace('<answer>', '')

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_answer_tag(lower(str(text))))))

# def normalize(text):
#     text = str(text).strip('<Answer>').lower().strip(string.punctuation)
#     return text

def clean_answer(df):
    df['answer'] = df['answer'].apply(normalize)
    df['Org Answer'] = df['Org Answer'].apply(normalize)
    df['R1 Answer'] = df['R1 Answer'].apply(normalize)
    df['R2 Answer'] = df['R2 Answer'].apply(normalize)
    return df

def compute_rouge_score(col1, col2):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    all_scores = []
    for ans1, ans2 in zip(col1, col2):
        score = scorer.score(ans1, ans2)
        all_scores.append(score['rouge1'].fmeasure)
    return all_scores


def main():
    ans_dir = 'answer'
    output_dir = 'rouge'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(5):
        filename = 'augment_fold_{:d}.csv'.format(i+1)
        print(filename)
        aug_df = pd.read_csv(os.path.join(ans_dir, filename))
        clean_aug_df = clean_answer(aug_df)
        r1_ans_score = compute_rouge_score(clean_aug_df['answer'], clean_aug_df['R1 Answer'])
        r1_org_score = compute_rouge_score(clean_aug_df['Org Answer'], clean_aug_df['R1 Answer'])
        r2_ans_score = compute_rouge_score(clean_aug_df['answer'], clean_aug_df['R2 Answer'])
        r2_org_score = compute_rouge_score(clean_aug_df['Org Answer'], clean_aug_df['R2 Answer'])
        clean_aug_df['r1_ans_score'] = r1_ans_score
        clean_aug_df['r1_org_score'] = r1_org_score
        clean_aug_df['r2_ans_score'] = r2_ans_score
        clean_aug_df['r2_org_score'] = r2_org_score
        output_path = os.path.join(output_dir, filename)
        clean_aug_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()