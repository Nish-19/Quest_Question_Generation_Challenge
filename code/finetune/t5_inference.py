import sys
import wandb, os
from collections import defaultdict
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneT5'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# load dataset
def get_parallel_corpus(ip_df, story_df):
    # hash stories and sections
    story_sec_hash = defaultdict(dict)
    for i, row in story_df.iterrows():
        story_sec_hash[row['source_title']][row['cor_section']] = row['text']
    
    story, answer, question = [], [], []
    for i, row in ip_df.iterrows():
        sec_nums = row['cor_section'].split(',')
        story_str = ''
        for sec_num in sec_nums:
            story_str += story_sec_hash[row['source_title']][int(sec_num)]
        story.append(story_str)
        answer.append(row['answer'])
        question.append(row['question'])
    
    return story, answer, question

# Constrcut t5 input 
def construct_t5_input(story, answer):
    inps = []
    prefix = 'Generate question from story and answer: '
    for stry, ans in zip(story, answer):
        t5_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
        inps.append(t5_input)
    return inps

def get_t5_encoding(t5_inputs, answer):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
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


# %%
story_file = '../../data/original/source_texts.csv'
story_df = pd.read_csv(story_file)
# Train-Val split
train_file = '../../data/train_val_split_csv/train.csv'
train_df = pd.read_csv(train_file)
val_file = '../../data/train_val_split_csv/val.csv'
val_df = pd.read_csv(val_file)

train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df)

# %%
train_inps = construct_t5_input(train_story, train_answer)
val_inps = construct_t5_input(val_story, val_answer)

# %%
train_input_ids, train_attention_mask, train_labels = get_t5_encoding(train_inps, train_question)
val_input_ids, val_attention_mask, val_labels = get_t5_encoding(val_inps, val_question)
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
model = T5ForConditionalGeneration.from_pretrained('./Checkpoints').to(device)
print('Successfully loaded saved checkpoint!')

# %%
# Generate from saved model
def get_generation(model, val_dataloader):
    val_outputs = []
    for step, batch in enumerate(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        generation = model.generate(val_input_ids)
        for gen in generation:
            val_outputs.append(gen)
    return val_outputs

def get_preds(generated_tokens):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

print('Beginning Generation')
val_outputs = get_generation(model, valid_dataloader)
print('Done Generating!')
print('Beginning Decoding')
val_preds = get_preds(val_outputs)
print('Done Decoding!')
preds_df = pd.DataFrame()
preds_df['predictions'] = val_preds
preds_df.to_excel('predictions.xlsx', index=False)
print(val_preds)