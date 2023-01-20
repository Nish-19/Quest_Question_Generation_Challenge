'''
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune.py
'''

import pathlib, sys
from typing import List
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import TRLConfig

import evaluate

# import torch
# torch.cuda.set_per_process_memory_fraction(0.8, 0)
# torch.cuda.set_per_process_memory_fraction(0.8, 1)
# torch.cuda.set_per_process_memory_fraction(0.8, 2)
# torch.cuda.set_per_process_memory_fraction(0.8, 3)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import sys
sys.path.append('/mnt/home/Quest_Question_Generation_Challenge/code/utils')
sys.path.append('/mnt/home/Quest_Question_Generation_Challenge/code/finetune/T5')
from compute_eval_metric import normalize
from t5_finetune import get_parallel_corpus, construct_t5_input

from pdb import set_trace


config_path = pathlib.Path(__file__).parent / "ppo_configs.yml"
config = TRLConfig.load_yaml(config_path)
bleurt = evaluate.load('bleurt', 'bleurt-20') # Use BLUERT as the evaluation metric

def get_data():
    story_file = '../../../data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = '../../../data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    val_file = '../../../data/train_val_split_csv/val.csv'
    val_df = pd.read_csv(val_file)
    train_story, train_answer, train_question, train_question_type = get_parallel_corpus(train_df, story_df)
    val_story, val_answer, val_question, val_question_type = get_parallel_corpus(val_df, story_df)
    # Construct T5 input
    train_inps = construct_t5_input(train_story, train_answer, train_question_type)
    val_inps = construct_t5_input(val_story, val_answer, val_question_type)
    return train_inps, train_question, val_inps, val_question

def truncate_input_and_discard_too_long(tokenizer, inps, questions, max_length):
    """
    Truncate inputs and discard those that are too long
    """
    new_inps = []
    new_questions = []
    for i in tqdm(range(len(inps))):
        tokenized_ids_original = tokenizer(inps[i])["input_ids"]
        if len(tokenized_ids_original) <= max_length:
            tokenized_ids = tokenizer(inps[i], truncation=True, max_length=max_length)["input_ids"]
            new_inps.append(tokenizer.decode(tokenized_ids, skip_special_tokens=True))
            new_questions.append(questions[i])
    return new_inps, new_questions

def map_inputs_to_questions(inps, questions):
    inputs_to_questions = {}
    for i in tqdm(range(len(inps))):
        inputs_to_questions[inps[i].strip()] = questions[i]
    return inputs_to_questions

if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        outputs = [normalize(t) for t in outputs]
        references = [normalize(inputs_to_questions[i.strip()]) for i in prompts]
        return bleurt.compute(predictions=outputs, references=references)["scores"]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)

    # Get data
    max_length = config.train.seq_length
    target_length = config.method.gen_kwargs["max_new_tokens"]
    train_inps, train_question, val_inps, val_question = get_data()
    print(f"Number of training examples: {len(train_inps)}")
    print(f"Number of validation examples: {len(val_inps)}")
    # Truncate inputs and discard those that are too long
    train_inps, train_questions = truncate_input_and_discard_too_long(
        tokenizer, train_inps, train_question, max_length)
    val_inps, val_questions = truncate_input_and_discard_too_long(
        tokenizer, val_inps, val_question, max_length)
    print(f"Number of training examples after truncation: {len(train_inps)}")
    print(f"Number of validation examples after truncation: {len(val_inps)}")

    # Make dictionary of prompts and labels to use for reward function
    inputs_to_questions_train = map_inputs_to_questions(train_inps, train_question)
    inputs_to_questions_val = map_inputs_to_questions(val_inps, val_question)
    inputs_to_questions_train.update(inputs_to_questions_val)
    inputs_to_questions = inputs_to_questions_train

    trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=train_inps,
        eval_prompts=val_inps,
        config=config,
    )