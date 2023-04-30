"""
Prepare dataset folds from the local train split to train our best flanT5 model to get its top 100 prediction questions for each sample. We'll use these top 100 questions for each sample to train our score prediction model.

python -m code.score_prediction.prepare_dataset_qg_model

Only for first creation:
python -m code.score_prediction.prepare_dataset_qg_model --first_creation
"""

import os
import argparse
import json
import pathlib
import numpy as np
import pandas as pd
import hashlib
from sklearn.model_selection import KFold

from code.utils.create_dataset_split import load_df, save_csv, save_json, check_data_error, get_stats


RAW_DIR = "./data/"
DATA_HASH_PATH = "./code/utils/data_hash_score_prediction_qg_model.json"
# original seed used for the original train-val fold we've been reporting results on
SEED = 21


def add_params():
    parser = argparse.ArgumentParser(description='prepare dataset for qg challenge')
    parser.add_argument('--first_creation', action='store_true', help='Use an extra dense layer on CLS')
    params = parser.parse_args()
    
    return params


def create_cross_val_folds(df, table, args):
    # Ensure stories in validation set are not in training set, so split across stories
    stories = df['source_title'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    print(kf)
    for fold, (train_index, val_index) in enumerate(kf.split(stories)):
        # Create train, val split
        print(f"\nCreating fold {fold}:")
        train_stories = stories[train_index]
        val_stories = stories[val_index]
        df_train = df[df['source_title'].isin(train_stories)]
        df_val = df[df['source_title'].isin(val_stories)]
        # Shuffle dataframes, although stories are shuffles before creating 5 CV folds, stories in each fold are not shuffled
        df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)
        # Print stats
        get_stats(SEED, train_stories, val_stories, df_train, df_val, df, stories)

        if( args.first_creation ):
            # Create data hash if fold created for the first time
            table[f"fold_{fold}"] = {}
            table[f"fold_{fold}"]["train"] = hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest() 
            table[f"fold_{fold}"]["val"] = hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest() 
        else:
            # Match data hash against original data hash
            assert table[f"fold_{fold}"]["train"] == hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest()
            assert table[f"fold_{fold}"]["val"] == hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest()
        
        save_data_splits(df_train, df_val, fold)


def save_data_splits(df_train, df_val, fold):
    # Save data splits
    json_dirname = os.path.join(RAW_DIR, f"score_prediction/qg_model/fold_{fold}/train_val_split_json")
    csv_dirname = os.path.join(RAW_DIR, f"score_prediction/qg_model/fold_{fold}/train_val_split_csv")
    pathlib.Path(json_dirname).mkdir(parents=True, exist_ok=True)
    pathlib.Path(csv_dirname).mkdir(parents=True, exist_ok=True)
    for split, name in [(df_train, "train"), (df_val, "val")]:
        save_json(split, name, json_dirname)
        save_csv(split, name, csv_dirname)
        
    
def main():
    args = add_params()
    # Load data
    filepath = os.path.join(RAW_DIR, "folds/seed_21/train_val_split_csv")
    df = load_df("train.csv", filepath)
    assert check_data_error(df) == False
    # Load hash of data if it exists to match data across users
    if( args.first_creation ):
        table = {} 
    else:
        with open(DATA_HASH_PATH) as f:
            table = json.load(f)
    # Create 5 fold cross validation splits
    np.random.seed(SEED)
    create_cross_val_folds(df, table, args)
    # Save data hash
    if( args.first_creation ):
        with open(DATA_HASH_PATH, "w") as f:
            json.dump(table, f, indent=4)


if __name__ == '__main__':
    main()