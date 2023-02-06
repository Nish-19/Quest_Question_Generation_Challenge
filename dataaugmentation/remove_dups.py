"""
python -m code.dataaugmentation.remove_dups \
    --folder "results/submission_leaderboard/" \
    --filename "best_submission_multiStrategy_umass_cur_best_02052023_0.90_1.00_16_ppl_reranked.csv" \
    --use_normalize

python -m code.dataaugmentation.remove_dups \
    --folder "augmentation/flan_t5_nischal/" \
    --filename "nucleus_flan_t5_large_0.95_1.20.csv"
"""

import os
import argparse

from code.utils.create_dataset_split import load_df, save_csv
from code.utils.compute_eval_metric import normalize

RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="train.csv", help="Folder name")
    parser.add_argument("--filename", type=str, default="train.csv", help="File name")
    parser.add_argument("--use_normalize", action='store_true', help='Use normalized generated question for deduplication')
    params = parser.parse_args()
    
    return params


def drop_duplicate_samples(df, use_normalize=False):
    # Count no of duplicate answers
    print("No of samples before deduplication: ", len(df))
    if( use_normalize ):
        df["generated_question_normalized"] = df["generated_question"].apply(lambda x: normalize(x))
        columns = ["pair_id", "generated_question_normalized"]
    else:
        columns = ["pair_id", "generated_question"]
    total_duplicates = len(df[df.duplicated(subset=columns, keep=False)])
    unique_duplicates = df.groupby(columns).size().gt(1).sum()
    print(f"No of duplicate samples removed = {total_duplicates - unique_duplicates} = {round((total_duplicates - unique_duplicates)/len(df) * 100, 2)}%")
    # Remove duplicates where pair_id and normalized generated question are the same
    df = df.drop_duplicates(subset=columns, keep="first")

    return df


def main():
    args = add_params()
    folder = os.path.join(RAW_DIR, args.folder)
    df = load_df(args.filename, folder)    
    df = drop_duplicate_samples(df, args.use_normalize)
    print("No of samples after deduplication: ", len(df))
    suffix = "_dedup_normalized.csv" if args.use_normalize else "_dedup.csv"
    filename = ".".join(args.filename.split(".")[:-1]) + suffix
    save_csv(df.reset_index(drop=True), filename, folder)


if __name__ == '__main__':
    main()