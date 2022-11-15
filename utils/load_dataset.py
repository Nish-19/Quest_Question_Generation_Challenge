"""
Load dataset splits for model training.
"""

import os
import json

from code.gpt3.prepare_dataset import load_stories


RAW_DIR = "./data/"


def load_dataset(data_folder="train_val_split_json", debug=False):
    dir_name = os.path.join(RAW_DIR, data_folder) 

    data = {}
    names = ["train", "val"]
    for name in names:
        filename = os.path.join(dir_name, "{}.json".format(name))
        with open(filename, "r") as f:
            data[name] = json.load(f)

    # Debug with less data
    if(debug):
        data["train"] = data["train"][0:4]
        data["val"] = data["val"][0:4]
        
    # Load stories
    story_map = load_stories()

    return data, story_map