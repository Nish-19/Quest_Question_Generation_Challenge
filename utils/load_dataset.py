"""
Load dataset splits for model training.
"""

import os
import json

from code.gpt3.prepare_dataset import load_stories
from code.utils.create_dataset_split import load_df

RAW_DIR = "./data/"


def load_dataset_score_prediction(data_folder="score_prediction", debug=False):
    data = {}
    dir_name = os.path.join(RAW_DIR, data_folder, "score_model", "train_val_test_split_json") 
    names = ["train", "val", "test"]
    for name in names:
        filepath = os.path.join(dir_name, "{}.json".format(name))
        with open(filepath, "r") as f:
            data[name] = json.load(f)
    # Load stories
    story_map = load_stories()

    # Debug with less data
    if(debug):
        data["train"] = data["train"][0:8]
        data["val"] = data["val"][0:8]
        data["test"] = data["test"][0:8]

    return data, story_map


def load_dataset(data_folder="folds", cross_val_fold=21, debug=False, augment=False, data_augment_folder="augmentation"):
    data = {}
    if( augment ):
        # Load external data as train set
        dir_data_augment = os.path.join(RAW_DIR, data_augment_folder, "external", "archive")
        df = load_df("train.csv", dir_data_augment)
        json_out = df.to_json(orient="records")
        data["train"] = json.loads(json_out)
        # Load original val data as val set
        dir_name = os.path.join(RAW_DIR, data_folder, f"seed_{cross_val_fold}", "train_val_split_json") 
        filepath = os.path.join(dir_name, "val.json")
        with open(filepath, "r") as f:
            data["val"] = json.load(f)
        # Load stories
        story_map_augment = load_stories(dir_data_augment, assertion=False)
        story_map_val = load_stories()
        story_map = {**story_map_augment, **story_map_val}

    else:
        dir_name = os.path.join(RAW_DIR, data_folder, f"seed_{cross_val_fold}", "train_val_split_json") 
        names = ["train", "val"]
        for name in names:
            filepath = os.path.join(dir_name, "{}.json".format(name))
            with open(filepath, "r") as f:
                data[name] = json.load(f)
        # Load stories
        story_map = load_stories()

    # Debug with less data
    if(debug):
        data["train"] = data["train"][0:8]
        data["val"] = data["val"][0:8]

    return data, story_map