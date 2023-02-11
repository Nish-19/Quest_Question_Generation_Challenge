import pandas as pd
import os
import random
import numpy as np
import torch
import re
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def load_df(filename, folder, nrows=None):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, nrows=nrows)
    df = df.fillna("")
    
    return df


def save_csv(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)


def set_random_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
    

def get_transformer_encoding(tokenizer, transformer_inputs, question, max_source_length=512, max_target_length=64):
    inp_encoding = tokenizer(transformer_inputs, padding='longest', 
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt"
                        )
    input_ids, attention_mask = inp_encoding.input_ids, inp_encoding.attention_mask

    target_encoding = tokenizer(question, padding='longest', 
                        max_length=max_target_length,
                        truncation=True,
                        return_tensors="pt"
                        )
    labels = target_encoding.input_ids
    # Don't compute loss for pad tokens
    labels[labels == tokenizer.pad_token_id] = -100
    
    return input_ids, attention_mask, labels


def get_dataloader(batch_size, dataset, datatype='train'):
    if datatype == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)


def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def construct_transformer_input(story, answer):
    # Construct transformer input 
    inps = []
    prefix = 'Generate question from context and answer: '
    for stry, ans in zip(story, answer):
        transformer_input = prefix + '\nContext: ' + stry + '\nAnswer: ' + ans
        inps.append(transformer_input)
    
    return inps


def get_parallel_corpus(ip_df, story_df, filetype='train'):
    # Hash stories and sections
    story_sec_hash = defaultdict(dict)
    for i, row in story_df.iterrows():
        story_sec_hash[row['source_title']][row['cor_section']] = clean_str(row['text'])
    
    story, answer, question = [], [], []
    for i, row in ip_df.iterrows():
        try:
            sec_nums = row['cor_section'].split(',')
        except AttributeError:
            sec_nums = [row['cor_section']]
        story_str = ''
        for sec_num in sec_nums:
            story_str += story_sec_hash[row['source_title']][int(sec_num)]
        story.append(story_str)
        answer.append(clean_str(row['answer']))
        if filetype == 'train':
            question.append(clean_str(row['question']))
        else:
            question.append('None')

    return story, answer, question