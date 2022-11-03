import pandas as pd
import os
import numpy as np
import json
import pathlib
import hashlib


SEED = 21
np.random.seed(SEED)
RAW_DIR = "./data/"
DATA_HASH_PATH = "./code/utils/data_hash.json"
NUM_CROSS_VAL_FOLDS = 5
# TRAIN_RATIO = 0.85 = 170/201 stories = 6005 / 6989 samples
# TRAIN_RATIO = 0.9 = 180/201 stories = 6276 / 6989 samples
TRAIN_RATIO = 0.85


def check_data_error(df):
    
    return df.isnull().values.any()


def load_df(filename, folder, nrows=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    df = df.fillna("")
    
    return df


def create_train_val_split(df, train_ratio=0.85):
    # Ensure stories in validation set are not in training set, so split across stories
    stories = df['source_title'].unique()
    n_train = int(len(stories)*train_ratio)
    # Shuffle stories
    np.random.shuffle(stories)
    # Split across stories
    train_stories = stories[0:n_train]
    val_stories = stories[n_train:]
    
    # Create train, val splits
    df_train = df[df['source_title'].isin(train_stories)]
    df_val = df[df['source_title'].isin(val_stories)]
    # Shuffle dataframes
    df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Print stats
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
    
    """
    # Create data hash if data splits created for the first time
    hash = {"train_val_split" : {}}
    hash["train_val_split"]["train"] = hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest() 
    hash["train_val_split"]["val"] = hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest() 
    with open(DATA_HASH_PATH, "w") as f:
        json.dump(hash, f, indent=4)
    """
    # Match data hash against original data hash
    with open(DATA_HASH_PATH) as f:
        hash = json.load(f)
    assert hash["train_val_split"]["train"] == hashlib.sha1(pd.util.hash_pandas_object(df_train).values).hexdigest()
    assert hash["train_val_split"]["val"] == hashlib.sha1(pd.util.hash_pandas_object(df_val).values).hexdigest()

    # Save data splits
    json_dirname = os.path.join(RAW_DIR, "train_val_split_json")
    csv_dirname = os.path.join(RAW_DIR, "train_val_split_csv")
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
    # Load data
    filepath = os.path.join(RAW_DIR, "original")
    df = load_df("train.csv", filepath)
    assert check_data_error(df) == False
    
    # Create data splits
    create_train_val_split(df, train_ratio=TRAIN_RATIO)
    

if __name__ == '__main__':
    main()