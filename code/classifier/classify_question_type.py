import os, re
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from pdb import set_trace

# %%
def clean_str(text):
  # Replace double quotes with single quotes
  # Remove non breaking spaces (\u00A0), etc
  text = re.sub(r"\s+", " ", text)
  return text.strip()
  
# Get data
def get_parallel_corpus(ip_df, story_df):
  # hash stories and sections
  story_sec_hash = defaultdict(dict)
  for i, row in story_df.iterrows():
    story_sec_hash[row['source_title']][row['cor_section']] = clean_str(row['text'])

  story, answer, question, question_type = [], [], [], []
  for i, row in ip_df.iterrows():
    sec_nums = row['cor_section'].split(',')
    story_str = ''
    for sec_num in sec_nums:
        story_str += story_sec_hash[row['source_title']][int(sec_num)]
    story.append(story_str)
    answer.append(clean_str(row['answer']))
    question.append(clean_str(row['question']))
    question_type.append(clean_str(row['attribute1']))
      
  return story, answer, question, question_type

# Constrcut BERT input 
def construct_input(story, answer):
  inps = []
  for stry, ans in zip(story, answer):
    t5_input = '[Story] ' + stry + ' [Answer] ' + ans 
    inps.append(t5_input)
  return inps

# Tokenization
def get_encoding(model_name, inputs, labels):
  label_map = {'character': 0, 'setting': 1, 'feeling': 2, 
                'action': 3, 'causal relationship': 4, 
                'outcome resolution': 5, 'prediction': 6}
  tokenizer = BertTokenizer.from_pretrained(model_name)
  max_source_length, max_target_length = 512, 128

  inp_encoding = tokenizer(inputs, padding='longest', 
                          max_length=max_source_length,
                          truncation=True,
                          return_tensors="pt"
                          )
  input_ids, attention_mask = inp_encoding.input_ids, inp_encoding.attention_mask
  labels = [label_map[l] for l in labels]
  return input_ids, attention_mask, torch.tensor(labels, dtype=torch.long)
  
# Pytorch Dataset
class ClassificationDataset(Dataset):
  def __init__(self, input_ids, attn_masks, labels):
    self.input_ids = input_ids
    self.attn_masks = attn_masks
    self.labels = labels
      
  def __getitem__(self, index):
    x = self.input_ids[index]
    y = self.attn_masks[index]
    z = self.labels[index]
      
    return {'input_ids': x, 'attention_mask': y, 'labels': z}
  
  def __len__(self):
    return len(self.input_ids)

# Dataset
def get_dataloader(batch_size, dataset, datatype='train'):
  if type == 'train':
    return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
  else:
    return DataLoader(dataset=dataset, batch_size = batch_size)

# %%
class BERTClassifier(pl.LightningModule):
  def __init__(self, model_name, training_dl=None, valid_dl=None, lr=3e-4, num_train_epochs=5, warmup_steps=50):
    super().__init__()
    self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)
    self.training_dataloader = training_dl
    self.valid_dataloader = valid_dl
    self.hparams.max_epochs = num_train_epochs
    self.hparams.num_train_epochs = num_train_epochs
    self.hparams.warmup_steps = warmup_steps
    self.hparams.lr = lr
    self.save_hyperparameters()
    
    # Metrics
    self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=7)
  
  def forward(self, input_ids, attention_mask, labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs
  
  def common_step(self, batch, batch_idx):
    return self(**batch)
  
  def training_step(self, batch, batch_idx):
    outputs = self.common_step(batch, batch_idx)     
    loss = outputs.loss
    # set_trace()
    acc = self.accuracy(outputs.logits, batch['labels'])
    # logs metrics for each training_step,
    # and the average across the epoch
    self.log("training_loss", loss)
    self.log("training_acc ", acc)
    return loss
  
  def validation_step(self, batch, batch_idx):
    outputs = self.common_step(batch, batch_idx)     
    loss = outputs.loss
    acc = self.accuracy(outputs.logits, batch['labels'])
    self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
    self.log("validation_acc ", acc , on_epoch=True, sync_dist=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    outputs = self.common_step(batch, batch_idx)     
    acc = self.accuracy(outputs.logits, batch['label'])
    return outputs.loss
  
  def configure_optimizers(self):
    # create optimizer
    optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
    # create learning rate scheduler
    with open('debug.txt', 'w') as outfile:
      print('In optmizer', file=outfile)
      print(self.hparams.lr, file=outfile)
      print(self.hparams.num_train_epochs, file=outfile)
      print(self.hparams.warmup_steps, file=outfile)

    num_train_optimization_steps = self.hparams.num_train_epochs * len(self.training_dataloader)
    lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=self.hparams.warmup_steps,
                                                num_training_steps=num_train_optimization_steps),
                    'name': 'learning_rate',
                    'interval':'step',
                    'frequency': 1}
      
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
  
  def train_dataloader(self):
    return self.training_dataloader

  def val_dataloader(self):
    return self.valid_dataloader
  
# %%
def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for training the BERT Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning Rate for training the BERT Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=5, help="Total Number of Epochs")
    parser.add_argument("-D", "--num_devices", type=int, default=1, help="Devices used for training")
    parser.add_argument("-M", "--model_name", type=str, default="distilbert-base-uncased", help="Variant of the BERT model for finetuning")
    params = parser.parse_args()
    
    return params
  
  
# %%
if __name__ == '__main__':
    args = add_params()
    run_name  = f'{args.model_name}.ep-{args.num_epochs}.bs-{args.batch_size}.lr-{args.learning_rate}.Ndevices-{args.num_devices}'

    story_file = '../../data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = '../../data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    val_file = '../../data/train_val_split_csv/val.csv'
    val_df = pd.read_csv(val_file)

    train_story, train_answer, _, train_labels = get_parallel_corpus(train_df, story_df)
    val_story, val_answer, _, val_labels = get_parallel_corpus(val_df, story_df)
    # set_trace()
    
    train_inps = construct_input(train_story, train_answer)
    val_inps = construct_input(val_story, val_answer)

    train_input_ids, train_attention_mask, train_labels = get_encoding(args.model_name, train_inps, train_labels)
    val_input_ids, val_attention_mask, val_labels = get_encoding(args.model_name, val_inps, val_labels)
    print('Tokenized Data!')

    train_dataset = ClassificationDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = ClassificationDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')

    batch_size = args.batch_size
    training_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')
    # set_trace()

    max_epochs = args.num_epochs
    model = BERTClassifier(model_name = args.model_name, training_dl=training_dataloader, 
                          valid_dl=valid_dataloader, num_train_epochs=max_epochs, 
                          lr=args.learning_rate)

    # Trainig code
    wandb_logger = WandbLogger(name=run_name, project='Quest_Gen_Challenge')

    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    save_directory = os.path.join('Checkpoints', run_name)
    save_checkpoint =  ModelCheckpoint(dirpath=save_directory, monitor='validation_loss', save_top_k=1)

    trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                    default_root_dir=save_directory, 
                    logger=wandb_logger,
                    max_epochs=max_epochs,
                    callbacks=[early_stop_callback, lr_monitor, save_checkpoint],
                    # strategy = DDPStrategy(find_unused_parameters=False)
                    # strategy = 'deepspeed_stage_3', precision=16
                    strategy = DeepSpeedStrategy(
                                    stage=3,
                                    offload_optimizer=True,
                                    offload_parameters=True,
                                    )
                    )

    trainer.fit(model)

    # if not os.path.exists(save_directory):
    #     os.mkdir(save_directory)
    # model.model.save_pretrained(save_directory)