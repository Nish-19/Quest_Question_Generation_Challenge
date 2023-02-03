"""
(new) module load cuda/10.1.243 && source /home/nigel_umass_edu/env/bin/activate && cd /work/nigel_umass_edu/code/qg_challenge
(old) module load cuda/10.1.243 && source /home/nigel_umass_edu/env/bin/activate && cd /home/nigel_umass_edu/qg_challenge

Run following command in virtual environment with tensorflow:

Score model:

python -m code.utils.compute_eval_metric \
    --eval_folder results/score_prediction/score_model/train_val_test_split_csv \
    --eval_filename QGCHAL-158_20230202-114840.csv \
    --batch_size 128 \
    --score_model


Finetuned flan-T5 for question generation:
(beam search with beam size 5)
(flan-t5-large) QGCHAL-54_20230130-023538_beam_search.csv = 0.4868478372057037
(flan-t5-base) QGCHAL-55_20230130-023220_beam_search.csv = 0.4764932113873765
(flan-t5-large + augment) QGCHAL-79_20230130-025423_beam_search.csv = 0.4827133711366876
(flan-t5-base + augment) QGCHAL-80_20230130-023306_beam_search.csv = 0.47534446515960666

(contrastive search with alpha=0.6 and top-k=4)
QGCHAL-54_20230130-050419_contrastive_search.csv = 0.6110344887748966
QGCHAL-55_20230130-045631_contrastive_search.csv = 0.5983508246140631
QGCHAL-79_20230130-050349_contrastive_search.csv = 0.6183827544706382
QGCHAL-80_20230130-045623_contrastive_search.csv = 0.598290389343127


Finetuned results:
python -m code.utils.compute_eval_metric \
    --eval_folder train_val_split_csv \
    --eval_filename curie_ft-umass-amherst_curie-train-2022-11-03-00-04-39_20221102-210508.csv \
    --batch_size 64
BLEURT:  0.49320319778005767
BLEURT:  0.5448525935956617 (without normalization)


python -m code.utils.compute_eval_metric \
    --eval_folder train_val_split_csv \
    --eval_filename code-davinci-002_20221212-070223.csv \
    --batch_size 128

GPT-3 Curie:
zero shot = 0.30488806867563145
incontext all = 0.29506806673376057
incontext random = 0.2970815305105918 (need better prompt design for stories demarcation)


Codex:
zero shot = 0.38967791111852096
incontext all without pack_max = 0.47531837670935123
incontext random without pack_max = 0.48398684873056363
incontext all with pack_max = 0.4813110683779649
incontext random with pack_max = 0.4757306356043593
zero shot (top-10, temp=0.99) = 0.5042377344627933
incontext augment (top-10, temp=0.99) = 0.6563515267448454
incontext augment (top-10, temp=0.7) = 0.6752303581291098


GPT-3 Davinci:
zero shot (old-002) = 0.3788071376294261
zero shot (new-003) = 0.4274963186069475


GPT2 finetuned with CLM loss on question only:
python -m code.utils.compute_eval_metric \
    --eval_folder gpt2/train_val_split_csv \
    --eval_filename crisp-oath-102_20221110-184440_top_k_sampling.csv \
    --batch_size 64

Greedy: 0.41023254576252727
Beam search: 0.4261451663403976
top_k_sampling: 
nucleus_sampling:
nucleus_sampling_with_top_k: 0.3559514638187924

GPT2 finetuned with CLM loss on all tokens:
python -m code.utils.compute_eval_metric \
    --eval_folder gpt2/train_val_split_csv \
    --eval_filename different-donkey-103_20221110-184450_top_k_sampling.csv \
    --batch_size 64

Greedy: 
Beam search: 0.4375158186047906
top_k_sampling:
nucleus_sampling:
nucleus_sampling_with_top_k: 0.37089890807988196
"""

import evaluate
import string
import re
import argparse
from tqdm import tqdm

from code.utils.create_dataset_split import load_df, RAW_DIR, save_csv
import os


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename with timestamp and .csv extension")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size for BLEURT model")
    parser.add_argument('--score_model', action='store_true', help='Output format of score model')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
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


def grade_score_without_batching(df, bleurt):
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


def grade_score_with_batching(df, bleurt, args):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions named function argument must match questions generated
    if( args.score_model ):
        # Rename columns if output file is from score/ranker model
        df = df.rename(columns={"generated_question_original": "generated_question", "question_original": "question"})

    df['question'] = df['question'].apply(normalize)
    df['generated_question'] = df['generated_question'].apply(normalize)

    ref_q = df['question'].tolist()
    gen_q = df['generated_question'].tolist()

    scores = []
    num_batches = ceildiv(len(ref_q), args.batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, args.batch_size), get_batch(gen_q, args.batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores


def keep_best_bleurt(df):
    # Keep only the question with the best BLEURT score
    return df.sort_values('bleurt_score', ascending=False).drop_duplicates(['pair_id'])


def compute_overall_bleurt_score(df_pred):
    df_pred = keep_best_bleurt(df_pred)
    print("Mean BLEURT grouped by question attribute type:\n", df_pred.groupby('attribute1')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question local vs summary:\n", df_pred.groupby('local_or_sum')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question explicit vs implicit:\n", df_pred.groupby('ex_or_im')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT over all samples: ", df_pred['bleurt_score'].mean())


def main():
    args = add_params()
    
    # Load BLEURT metric
    bleurt = evaluate.load('bleurt', 'bleurt-20')
    
    # Load question generations
    folder = os.path.join(RAW_DIR, args.eval_folder)
    nrows = 5 if args.debug else None
    df_pred = load_df(args.eval_filename, folder, nrows)
    
    # Non batching method
    #bleurt_list = grade_score_without_batching(df_pred, bleurt)
    
    # Batching method
    bleurt_scores = grade_score_with_batching(df_pred, bleurt, args)
    df_pred['bleurt_score'] = bleurt_scores
    df_pred['bleurt_score'] = df_pred['bleurt_score'].astype(float)

    compute_overall_bleurt_score(df_pred)

    # Save file with BLEURT scores
    save_csv(df_pred, "{}_bleurt".format(args.eval_filename.split(".")[0]), folder)


if __name__ == '__main__':
    main()