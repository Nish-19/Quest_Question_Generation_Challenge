'''
python -m code.finetune.Bart.bart_inference \
-N bart_base -M facebook/bart-base \
-BS -NB 5
'''

import argparse
import re
import wandb, os
from collections import defaultdict
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from code.utils.create_dataset_split import RAW_DIR, save_csv
from code.finetune.Bart.bart_finetune import FinetuneTransformer

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneBart'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# load dataset
def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_parallel_corpus(ip_df, story_df):
    # hash stories and sections
    story_sec_hash = defaultdict(dict)
    for i, row in story_df.iterrows():
        story_sec_hash[row['source_title']][row['cor_section']] = clean_str(row['text'])
    
    story, answer, question = [], [], []
    for i, row in ip_df.iterrows():
        sec_nums = row['cor_section'].split(',')
        story_str = ''
        for sec_num in sec_nums:
            story_str += story_sec_hash[row['source_title']][int(sec_num)]
        story.append(story_str)
        answer.append(clean_str(row['answer']))
        question.append(clean_str(row['question']))
    
    return story, answer, question

# Constrcut t5 input 
def construct_input(story, answer):
    inps = []
    prefix = 'Generate question from story and answer: '
    for stry, ans in zip(story, answer):
        transformer_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
        inps.append(transformer_input)
    return inps

def get_encoding(model_name, t5_inputs, answer):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 512, 128

    inp_encoding = tokenizer(t5_inputs, padding='longest', 
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt"
                    )
    input_ids, attention_mask = inp_encoding.input_ids, inp_encoding.attention_mask

    target_encoding = tokenizer(answer, padding='longest', 
                        max_length=max_target_length,
                        truncation=True,
                        return_tensors="pt"
                    )
    
    labels = target_encoding.input_ids

    # 0 loss for pad tokens
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, labels

class FairyDataset(Dataset):
    def __init__(self, input_ids, attn_masks, labels):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        
    def __getitem__(self, index):
        x = self.input_ids[index]
        y = self.attn_masks[index]
        z = self.labels[index]
        
        return {'input_ids': x, 'attention_mask': y, 'labels':z}
    
    def __len__(self):
        return len(self.input_ids)

def get_dataloader(batch_size, dataset, datatype='train'):
    if type == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)

# Generate from saved model
def get_generation(model, val_dataloader, force_words_ids, beam_search=True, num_beams=3):
    val_outputs = []
    for step, batch in enumerate(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        # TODO: Force ? to occur in the sentence
        if beam_search:
            generation = model.generate(val_input_ids, force_words_ids=force_words_ids, 
                                        num_beams = num_beams, max_new_tokens=64)
        else:
            generation = model.generate(val_input_ids, max_new_tokens=64)
        for gen in generation:
            val_outputs.append(gen)
    return val_outputs

def get_preds(model_name, generated_tokens):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for training the Transformer Model")
    parser.add_argument("-N", "--run_name", type=str, default="bart_base", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-M", "--model_name", default="bart-base", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-F", "--eval_folder", type=str, default="train_val_split_csv", help="Evaluation Folder where output is saved")
    parser.add_argument('-BS', '--beam_search', action=argparse.BooleanOptionalAction, help='Enables beam search')
    parser.add_argument('-NB', '--num_of_beams', type=int, default=3, help="Number of beams for decoding")
    params = parser.parse_args()
    
    return params


# %%
if __name__=='__main__':
    args = add_params()

    story_file = './data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = './data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    val_file = './data/train_val_split_csv/val.csv'
    val_df = pd.read_csv(val_file)

    train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
    val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df)

    # %%
    train_inps = construct_input(train_story, train_answer)
    val_inps = construct_input(val_story, val_answer)

    # %%
    train_input_ids, train_attention_mask, train_labels = get_encoding(args.model_name, train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_encoding(args.model_name, val_inps, val_question)
    print('Tokenized Data!')

    # %%
    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')

    # %%
    batch_size = 8
    train_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    # %%
    # Load the Generative Head 
    # search for ckpt file
    search_dir = os.path.join('./code/finetune/Checkpoints', args.run_name)
    for file in os.listdir(search_dir):
        ckpt_file = os.path.join(search_dir, file)
    model = FinetuneTransformer.load_from_checkpoint(ckpt_file).model.to(device)
    print('Successfully loaded the saved checkpoint!')

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    force_tokens = ['?']
    force_words_ids = tokenizer(force_tokens, add_special_tokens=False).input_ids

    print('Begining Generation')
    val_outputs = get_generation(model, valid_dataloader, force_words_ids, args.beam_search, args.num_of_beams)
    print('Done Generating!')
    print('Begining Decoding')
    val_preds = get_preds(args.model_name, val_outputs)
    print('Done Decoding!')
    preds_df = pd.DataFrame()
    preds_df['attribute1'] = val_df['attribute1']
    preds_df['local_or_sum'] = val_df['local_or_sum']
    preds_df['ex_or_im'] = val_df['ex_or_im']
    preds_df['prompt'] = val_inps
    preds_df['question'] = val_question
    preds_df['generated_question'] = val_preds
    output_path = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_csv(preds_df, args.run_name, output_path)