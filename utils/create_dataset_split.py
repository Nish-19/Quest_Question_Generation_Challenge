"""
python -m code.utils.create_dataset_split
"""

import pandas as pd
import os
import numpy as np
import json
import pathlib
import hashlib
import argparse


RAW_DIR = "./data/"
DATA_HASH_PATH = "./code/utils/data_hash.json"
# TRAIN_RATIO = 0.85 = 170/201 stories = 6005 / 6989 samples
# TRAIN_RATIO = 0.9 = 180/201 stories = 6276 / 6989 samples
TRAIN_RATIO = 0.85


def add_params():
    parser = argparse.ArgumentParser(description='prepare dataset for qg challenge')
    parser.add_argument('--first_creation', action='store_true', help='Use an extra dense layer on CLS')
    params = parser.parse_args()
    
    return params
    

def check_data_error(df):
    
    return df.isnull().values.any()


def load_df(filename, folder, nrows=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    df = df.fillna("")
    
    return df


def create_train_val_split(df, seed, args, table, train_ratio=0.85):
    # Ensure stories in validation set are not in training set, so split across stories
    stories = df['source_title'].unique()
    n_train = int(len(stories)*train_ratio)
    # Shuffle and split across stories
    np.random.shuffle(stories)
    train_stories = stories[0:n_train]
    val_stories = stories[n_train:]
    
    # Create train, val splits
    df_train = df[df['source_title'].isin(train_stories)]
    df_val = df[df['source_title'].isin(val_stories)]
    # Shuffle dataframes
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Print stats
    print(f"\nCreated fold with seed: {seed}")
    print("=== Stories ===")
    print("Num train stories: ", len(train_stories))
    print("Num val stories: ", len(val_stories))
    print("Percent train stories: ", len(train_stories)/len(stories))
    print("Percent val stories: ", len(val_stories)/len(stories))
    print("=== Samples ===")
    print("Num train samples: ", len(df_train))
    print("Num val samples: ", len(df_val))
    print("Percent train samples: ", len(df_train)/len(df))
    print("Percent val samples: ", len(df_val)/len(df))
    
    if( args.first_creation ):
        # Create data hash if data splits created for the first time
        table[f"seed_{seed}"] = {}
        table[f"seed_{seed}"]["train"] = hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest() 
        table[f"seed_{seed}"]["val"] = hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest() 
    else:
        # Match data hash against original data hash
        assert table[f"seed_{seed}"]["train"] == hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest()
        assert table[f"seed_{seed}"]["val"] == hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest()

    return df_train, df_val


def save_data_splits(df_train, df_val, seed):
    # Save data splits
    json_dirname = os.path.join(RAW_DIR, f"folds/seed_{seed}/train_val_split_json")
    csv_dirname = os.path.join(RAW_DIR, f"folds/seed_{seed}/train_val_split_csv")
    pathlib.Path(json_dirname).mkdir(parents=True, exist_ok=True)
    pathlib.Path(csv_dirname).mkdir(parents=True, exist_ok=True)
    for split, name in [(df_train, "train"), (df_val, "val")]:
        save_json(split, name, json_dirname)
        save_csv(split, name, csv_dirname)


def create_train_val_test_split(df):
    # Ensure stories in validation and test set are not in training set, so split across stories
    pass


def create_cross_validation_split(df):
    # Create folds across stories not samples, to ensure stories are not seen across folds
    pass


def save_csv(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)


def save_json(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".json")
    json_out = df.to_json(orient="records")
    dataset = json.loads(json_out)
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=2)  


def main():
    args = add_params()
    # Load data
    filepath = os.path.join(RAW_DIR, "original")
    df = load_df("train.csv", filepath)
    assert check_data_error(df) == False
    # Load hash of data if it exists to match data across users
    if( args.first_creation ):
        table = {} 
    else:
        with open(DATA_HASH_PATH) as f:
            table = json.load(f)
    # Create data splits with different random seeds
    # Seed 21 was the original seed we used to create the data splits
    for seed in [21, 0, 1, 2, 3]:
        np.random.seed(seed)
        df_train, df_val = create_train_val_split(df, seed, args, table, train_ratio=TRAIN_RATIO)
        save_data_splits(df_train, df_val, seed)
    # Save data hash
    if( args.first_creation ):
        with open(DATA_HASH_PATH, "w") as f:
            json.dump(table, f, indent=4)
    

if __name__ == '__main__':
    main()