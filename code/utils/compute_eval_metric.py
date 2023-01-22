"""
Run following command in virtual environment with tensorflow:

srun --pty -p gpu-long --mem=16000 --partition=gypsum-rtx8000 --gres=gpu:1 bash
module load cuda/10.1.243 && source env/bin/activate && cd qg_challenge/qg_challenge

Finetuned results:
python -m code.utils.compute_eval_metric \
    --eval_folder train_val_split_csv \
    --eval_filename curie_ft-umass-amherst_curie-train-2022-11-03-00-04-39_20221102-210508.csv \
    --batch_size 64
BLEURT:  0.49320319778005767
BLEURT:  0.5448525935956617 (without normalization)

Zero shot prompting results:
python -m code.utils.compute_eval_metric \
    --eval_folder train_val_split_csv \
    --eval_filename text-curie-001_20221103-030617.csv \
    --batch_size 64
BLEURT:  0.31435113114766716
BLEURT:  0.38770477067891174 (without normalization)
"""
import sys
import numpy as np
import pandas as pd
import evaluate
import string
import re
import argparse
from tqdm import tqdm

from code.utils.create_dataset_split import load_df, RAW_DIR, save_csv
import os


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-EFD', "--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument('-EFN', "--eval_filename", type=str, default="val.csv", help="Evaluation filename with timestamp and .csv extension")
    parser.add_argument('-T', "--type", type=str, default="Q", help="Q-Question, A-Answer")
    parser.add_argument('-B', "--batch_size", type=int, default=64, help="Evaluation batch size for BLEURT model")
    params = parser.parse_args()
    
    return params


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


def grade_score(df, bleurt):
    nls = []
    for curit, (q, gq) in enumerate(zip(df['question'], df['generated_question'])):
        result = bleurt.compute(predictions=[normalize(gq)], references=[normalize(q)])
        nls.append(result)
    return nls


def get_batch(iterable, n=1):
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ceildiv(a, b):
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    return -(a // -b)


def grade_score_with_batching_question(df, bleurt, batch_size=64):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions must match questions generated
    df['question'] = df['question'].apply(normalize)
    df['generated_question'] = df['generated_question'].apply(normalize)

    ref_q = df['question'].tolist()
    gen_q = df['generated_question'].tolist()

    scores = []
    num_batches = ceildiv(len(ref_q), batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, batch_size), get_batch(gen_q, batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores

def grade_score_with_batching_answer(df, bleurt, batch_size=64):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions must match questions generated
    df['answer'] = df['answer'].apply(normalize)
    df['generated_answer'] = df['generated_answer'].apply(normalize)

    ref_q = df['answer'].tolist()
    gen_q = df['generated_answer'].tolist()

    scores = []
    num_batches = ceildiv(len(ref_q), batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, batch_size), get_batch(gen_q, batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores

def report_pairidwise_preds(df_pred):
    attr, ex_or_im, loc_or_sum = [], [], []
    grp_wise_max_bluert = []
    grp_pair_ids = df_pred.groupby('pair_id')
    for grp_full in grp_pair_ids:
        grp = grp_full[1]
        max_score = max(grp['bleurt_score'])
        # print('max_score:', max_score)
        grp_wise_max_bluert.append(max_score)
        attr.append(grp['attribute1'].tolist()[0])
        ex_or_im.append(grp['ex_or_im'].tolist()[0])
        loc_or_sum.append(grp['local_or_sum'].tolist()[0])
    
    reduced_df = pd.DataFrame()
    reduced_df['attribute1'] = attr
    reduced_df['local_or_sum'] = loc_or_sum
    reduced_df['ex_or_im'] = ex_or_im
    reduced_df['bleurt_score'] = grp_wise_max_bluert
    print('Mean BLEURT score:', np.mean(grp_wise_max_bluert))
    print("Mean BLEURT grouped by question attribute type:\n", reduced_df.groupby('attribute1')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question local vs summary:\n", reduced_df.groupby('local_or_sum')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question explicit vs implicit:\n", reduced_df.groupby('ex_or_im')['bleurt_score'].agg(['mean', 'count']))


def main():
    args = add_params()
    
    # Load BLUERT metric
    bleurt = evaluate.load('bleurt', 'bleurt-20')
    
    # Load question generations
    folder = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    df_pred = load_df(args.eval_filename, folder)#, nrows=10)
    
    # Non batching method
    #bleurt_list = grade_score(df_pred, bleurt)
    #print("BLEURT: ", np.mean([x["scores"][0] for x in bleurt_list]))
    
    # Batching method
    if args.type == 'Q':
        bleurt_scores = grade_score_with_batching_question(df_pred, bleurt, args.batch_size)
    elif args.type == 'A':
        bleurt_scores = grade_score_with_batching_answer(df_pred, bleurt, args.batch_size)

    # Groupwise preds
    

    #print(bleurt_scores)
    # print("Mean BLEURT over all samples: ", np.mean(bleurt_scores))

    # Get average BLEURT scores per question type
    df_pred['bleurt_score'] = bleurt_scores
    df_pred['bleurt_score'] = df_pred['bleurt_score'].astype(float)

    report_pairidwise_preds(df_pred)

    # print("Mean BLEURT grouped by question attribute type:\n", df_pred.groupby('attribute1')['bleurt_score'].agg(['mean', 'count']))
    # print("Mean BLEURT grouped by question local vs summary:\n", df_pred.groupby('local_or_sum')['bleurt_score'].agg(['mean', 'count']))
    # print("Mean BLEURT grouped by question explicit vs implicit:\n", df_pred.groupby('ex_or_im')['bleurt_score'].agg(['mean', 'count']))

    # Save file with BLEURT scores
    file_name, file_ext = os.path.splitext(args.eval_filename)
    save_csv(df_pred, "{}_bluert".format(file_name), folder)


if __name__ == '__main__':
    main()