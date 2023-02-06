"""
python -m code.utils.create_dataset_split 
(with test split)
python -m code.utils.create_dataset_split --add_test_split

Only for first creation:
python -m code.utils.create_dataset_split --first_creation
(with test split)
python -m code.utils.create_dataset_split --first_creation --add_test_split
"""

import pandas as pd
import os
import numpy as np
import json
import pathlib
import hashlib
import argparse
import itertools


RAW_DIR = "./data/"


def add_params():
    parser = argparse.ArgumentParser(description='prepare dataset for qg challenge')
    parser.add_argument('--first_creation', action='store_true', help='Use an extra dense layer on CLS')
    parser.add_argument('--add_test_split', action='store_true', help='Create train-val-test split and not just train-val split')
    params = parser.parse_args()
    
    return params
    

def check_data_error(df):
    
    return df.isnull().values.any()


def load_df(filename, folder, nrows=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    df = df.fillna("")
    
    return df


def get_stats(seed, train_stories, val_stories, df_train, df_val, df, stories, test_stories=None, df_test=None):
    # Print stats
    print(f"\nCreated fold with seed: {seed}")
    print("=== Stories ===")
    print("Num train stories: ", len(train_stories))
    print("Num val stories: ", len(val_stories))
    if( test_stories is not None ):
        print("Num test stories: ", len(test_stories))
    print("Percent train stories: ", len(train_stories)/len(stories))
    print("Percent val stories: ", len(val_stories)/len(stories))
    if( test_stories is not None ):
        print("Percent test stories: ", len(test_stories)/len(stories))
    print("=== Samples ===")
    print("Num train samples: ", len(df_train))
    print("Num val samples: ", len(df_val))
    if( test_stories is not None ):
        print("Num test samples: ", len(df_test))
    print("Percent train samples: ", len(df_train)/len(df))
    print("Percent val samples: ", len(df_val)/len(df))
    if( test_stories is not None ):
        print("Percent test samples: ", len(df_test)/len(df))


def create_data_split(df, seed, args, table, add_test_split=False):
    if( add_test_split ):
        train_ratio = 0.7
        val_ratio = 0.15
    else:
        # train_ratio = 0.85 = 170/201 stories = 6005 / 6989 samples
        train_ratio = 0.85
    
    # Ensure stories in validation (and test) set are not in training set, so split across stories
    stories = df['source_title'].unique()
    n_train = int(len(stories)*train_ratio)
    # Shuffle stories
    np.random.shuffle(stories)
    if( add_test_split ):
        n_val = int(len(stories)*val_ratio)
        train_stories = stories[0:n_train]
        val_stories = stories[n_train:n_train+n_val]
        test_stories = stories[n_train+n_val:]
    else:        
        train_stories = stories[0:n_train]
        val_stories = stories[n_train:]
        test_stories = None
    
    # Create train, val, test splits
    df_train = df[df['source_title'].isin(train_stories)]
    df_val = df[df['source_title'].isin(val_stories)]
    if( add_test_split ):
        df_test = df[df['source_title'].isin(test_stories)]
    else:
        df_test = None
    # Shuffle dataframes
    df_train = shuffle_df(df_train, seed)
    df_val = shuffle_df(df_val, seed)
    if( add_test_split ):
        df_test = shuffle_df(df_test, seed)
    # Print stats
    get_stats(seed, train_stories, val_stories, df_train, df_val, df, stories, test_stories, df_test)

    if( args.first_creation ):
        # Create data hash if data splits created for the first time
        table[f"seed_{seed}"] = {}
        table[f"seed_{seed}"]["train"] = hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest() 
        table[f"seed_{seed}"]["val"] = hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest() 
        if( add_test_split ):
            table[f"seed_{seed}"]["test"] = hashlib.sha1(pd.util.hash_pandas_object(df_test).values).hexdigest()
    else:
        # Match data hash against original data hash
        assert table[f"seed_{seed}"]["train"] == hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest()
        assert table[f"seed_{seed}"]["val"] == hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest()
        if( add_test_split ):
            assert table[f"seed_{seed}"]["test"] == hashlib.sha1(pd.util.hash_pandas_object(df_test).values).hexdigest()

    return df_train, df_val, df_test


def shuffle_df(df, seed):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df


def save_data_splits(df_train, df_val, seed, df_test=None):
    # Save data splits
    name = "train_val_test_split" if df_test is not None else "train_val_split"
    json_dirname = os.path.join(RAW_DIR, f"folds/seed_{seed}/{name}_json")
    csv_dirname = os.path.join(RAW_DIR, f"folds/seed_{seed}/{name}_csv")
    pathlib.Path(json_dirname).mkdir(parents=True, exist_ok=True)
    pathlib.Path(csv_dirname).mkdir(parents=True, exist_ok=True)
    splits = [(df_train, "train"), (df_val, "val"), (df_test, "test")] if df_test is not None else [(df_train, "train"), (df_val, "val")]
    for split, name in splits:
        save_json(split, name, json_dirname)
        save_csv(split, name, csv_dirname)


def save_csv(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)


def save_json(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".json")
    json_out = df.to_json(orient="records")
    dataset = json.loads(json_out)
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=2)  


def disjoint_stories_check(df_1, df_2):
    # Assert stories in df_1 and df_2 are disjoint
    if( (df_1 is None) or (df_2 is None) ):
        assert True
    else:
        assert len(set(df_1["source_title"].values).intersection(set(df_2["source_title"].values))) == 0


def main():
    args = add_params()
    # Load data
    filepath = os.path.join(RAW_DIR, "original")
    df = load_df("train.csv", filepath)
    assert check_data_error(df) == False
    # Load hash of data if it exists to match data across users
    data_hash_filename = "data_hash_train_val_test_split.json" if args.add_test_split else "data_hash_train_val_split.json"
    data_hash_path = os.path.join("code/utils", data_hash_filename)
    if( args.first_creation ):
        table = {} 
    else:
        with open(data_hash_path) as f:
            table = json.load(f)
    # Create data splits with different random seeds
    # Seed 21 was the original seed we used to create the data splits
    for seed in [21, 0, 1, 2, 3]:
        np.random.seed(seed)
        df_train, df_val, df_test = create_data_split(df, seed, args, table, args.add_test_split)
        save_data_splits(df_train, df_val, seed, df_test)
        # Check stories in train, val, and test are disjoint
        for pair in itertools.combinations([df_train, df_val, df_test], 2):
            disjoint_stories_check(pair[0], pair[1])
    # Save data hash
    if( args.first_creation ):
        with open(data_hash_path, "w") as f:
            json.dump(table, f, indent=4)
    

if __name__ == '__main__':
    main()