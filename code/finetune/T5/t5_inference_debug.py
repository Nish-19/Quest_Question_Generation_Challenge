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

# %%
story_file = 'Quest_Question_Generation_Challenge/data/original/source_texts.csv'
story_df = pd.read_csv(story_file)
# Train-Val split
train_file = 'Quest_Question_Generation_Challenge/data/train_val_split_csv/train.csv'
train_df = pd.read_csv(train_file)
val_file = 'Quest_Question_Generation_Challenge/data/train_val_split_csv/val.csv'
val_df = pd.read_csv(val_file)

train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df)

# %%
def get_stats(story, answer, question):
    print('Average story length:', statistics.mean([len(stry) for stry in story]))
    print('Average answer length:', statistics.mean([len(ans) for ans in answer]))
    print('Average question length:', statistics.mean([len(quest) for quest in question]))

# %%
# print stats
print('Train Set')
get_stats(train_story, train_answer, train_question)

print('Valid Set')
get_stats(val_story, val_answer, val_question)

# %%
# Constrcut t5 input 
def construct_t5_input(story, answer):
    inps = []
    prefix = 'Generate question from story and answer: '
    for stry, ans in zip(story, answer):
        t5_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
        inps.append(t5_input)
    return inps

# %%
train_inps = construct_t5_input(train_story, train_answer)
val_inps = construct_t5_input(val_story, val_answer)

# %%
def get_t5_encoding(t5_inputs, question):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    max_source_length, max_target_length = 512, 128

    inp_encoding = tokenizer(t5_inputs, padding='longest', 
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

    # 0 loss for pad tokens
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, labels

# %%
train_input_ids, train_attention_mask, train_labels = get_t5_encoding(train_inps, train_question)
val_input_ids, val_attention_mask, val_labels = get_t5_encoding(val_inps, val_question)
print('Tokenized Data!')

# %%
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

# %%
train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
print('Created Pytorch Dataset')

# %%
def get_dataloader(batch_size, dataset, datatype='train'):
    if type == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)

# %%
batch_size = 8
train_dataloader = get_dataloader(batch_size, train_dataset)
valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
print('Loaded Dataloader!')

# %%
class FinetuneT5(pl.LightningModule):
    def __init__(self, pretrain_source='t5-small', lr=5e-5, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_source)
        self.hparams.max_epochs = num_train_epochs
        self.hparams.num_train_epochs = num_train_epochs
        self.hparams.warmup_steps = warmup_steps
        self.hparams.lr = lr
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss
    
    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        with open('debug.txt', 'w') as outfile:
            print('In optmizer', file=outfile)
            print(self.hparams.lr, file=outfile)
            print(self.hparams.num_train_epochs, file=outfile)
            print(self.hparams.warmup_steps, file=outfile)

        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader


# %%
wandb.login()

# %%
max_epochs = 5
model = T5ForConditionalGeneration.from_pretrained('Quest_Question_Generation_Challenge/code/finetune/Checkpoints').to(device)
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