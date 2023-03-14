import os 
import json 
import pandas as pd 
from collections import Counter

def display_attr_stat(train_df, aug_df, to_add=False):
    train_attr_stats = dict(sorted(dict(Counter(train_df['attribute'])).items(), key= lambda item : item[1], reverse=True))
    aug_attr_stats = dict(sorted(dict(Counter(aug_df['attribute'])).items(), key= lambda item : item[1], reverse=True))
    attr_names = list(train_attr_stats.keys())
    train_attr_values = list(train_attr_stats.values())
    aug_attr_values = [aug_attr_stats[attr_name] for attr_name in attr_names]
    attr_df = pd.DataFrame()
    attr_df['Attr Name'] = attr_names
    attr_df['Train Count'] = train_attr_values
    attr_df['Aug Count'] = aug_attr_values
    if to_add:
        total_values = [tv + av for tv, av in zip(train_attr_values, aug_attr_values)]
        attr_df['Total Count'] = total_values
    print(attr_df)

def main():
    # NOTE: Aug df
    filter_dir = 'filter'
    filename = 'filter_aug.csv'
    aug_df = pd.read_csv(os.path.join(filter_dir, filename))
    print('Attribute Distribution')

    # NOTE: Train data
    data_path = '../data/FairytaleQA/train.json'
    train_data = []
    with open(data_path, 'r') as infile:
        for line in infile:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    # Display stats 
    display_attr_stat(train_df, aug_df, to_add=True)

    # NOTE: Type 0 (Dump only the augment data)
    org_data = []
    aug_df_cols = list(aug_df.columns)
    for i, row in aug_df.iterrows():
        vals = row.values
        aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
        # augment to the train_data
        org_data.append(aug_dict)

    # Store the augmented data
    store_dir = '../data/FairytaleQA'
    store_filename = 'prompt_only_data.json'
    with open(os.path.join(store_dir, store_filename), 'w') as f:
        for d in org_data:
            json.dump(d, f)
            f.write('\n')

    # # NOTE: Aug Type 1: Complete Data Augmentation 
    # org_data = train_data.copy()
    # aug_df_cols = list(aug_df.columns)
    # for i, row in aug_df.iterrows():
    #     vals = row.values
    #     aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
    #     # augment to the train_data
    #     org_data.append(aug_dict)
    
    # # Store the augmented data
    # store_dir = '../data/FairytaleQA'
    # store_filename = 'prompt_aug_full_train.json'
    # with open(os.path.join(store_dir, store_filename), 'w') as f:
    #     for d in org_data:
    #         json.dump(d, f)
    #         f.write('\n')

    # # NOTE: Aug Type 2: Augment only character, feeling, outcome resolution, setting and prediction
    # org_data = train_data.copy()
    # aug_df_cols = list(aug_df.columns)
    # aug_choices = ['character', 'feeling', 'outcome resolution', 'setting', 'prediction']
    # for i, row in aug_df.iterrows():
    #     if row['attribute'] in aug_choices:
    #         vals = row.values
    #         aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
    #         # augment to the train_data
    #         org_data.append(aug_dict)
    
    # # Store the augmented data
    # store_dir = '../data/FairytaleQA'
    # store_filename = 'prompt_aug_selective_train.json'
    # with open(os.path.join(store_dir, store_filename), 'w') as f:
    #     for d in org_data:
    #         json.dump(d, f)
    #         f.write('\n')

if __name__ == '__main__':
    main()

