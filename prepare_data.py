import os
import re
import json
from collections import defaultdict
import pandas as pd

def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Constrcut transformer input 
def construct_transformer_input(story, answer):
    inps = []
    prefix = 'Generate question from context and answer: '
    for stry, ans in zip(story, answer):
        transformer_input = prefix + '\nContext: ' + stry + '\nAnswer: ' + ans
        inps.append(transformer_input)
    return inps

# load dataset
def get_parallel_corpus(ip_df, story_df, filetype='train'):
    # hash stories and sections
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
            question.append(None)

    return story, answer, question

if __name__ == '__main__':

    with open('SETTINGS.json', 'r') as infile:
        json_file = json.load(infile)

    data_dir = json_file['RAW_DATA_DIR']
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    story_file = os.path.join(data_dir, 'source_texts.csv')

    # Load as dataframe
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    story_df = pd.read_csv(story_file)

    # Fetch information
    train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
    test_story, test_answer, test_question = get_parallel_corpus(test_df, story_df, filetype='test')

    # Prepare transformer inputs
    train_inps = construct_transformer_input(train_story, train_answer)
    test_inps = construct_transformer_input(test_story, test_answer)

    # Save the prepared inputs and outputs
    clean_train_df = pd.DataFrame()
    clean_train_df['Transformer Input'] = train_inps
    clean_train_df['Transformer Output'] = train_question

    clean_test_df = pd.DataFrame()
    clean_test_df['Transformer Input'] = test_inps

    # check if save directory exists 
    if not os.path.exists(json_file['CLEAN_DATA_DIR']):
        os.mkdir(json_file['CLEAN_DATA_DIR'])

    clean_train_df.to_csv(json_file['TRAIN_DATA_CLEAN_PATH'], index=False)
    clean_test_df.to_csv(json_file['TEST_DATA_CLEAN_PATH'], index=False)

    print('Prepared the raw data for training and inference')
    print('Cleaned Train File Path: {:s}'.format(json_file['TRAIN_DATA_CLEAN_PATH']))
    print('Cleaned Test File Path: {:s}'.format(json_file['TEST_DATA_CLEAN_PATH']))






    
