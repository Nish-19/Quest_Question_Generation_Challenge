# %%
'''
python -m code.finetune.finetune \
    -MT T \
    -MN t5-small \
    -N t5_small
'''

# %%
import sys
import re
import wandb, os
from collections import defaultdict
import argparse
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneTransformer'
wandb.login()

# %%
def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# load dataset
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

def get_stats(story, answer, question):
    print('Average story length:', statistics.mean([len(stry) for stry in story]))
    print('Average answer length:', statistics.mean([len(ans) for ans in answer]))
    print('Average question length:', statistics.mean([len(quest) for quest in question]))

# Constrcut transformer input 
def construct_transformer_input(story, answer, choice=1):
    inps = []
    if choice == 1:
        prefix = 'Generate question from story and answer: '
    elif choice == 2:
        prefix = 'Generate question: '
    else:
        prefix = ''
    for stry, ans in zip(story, answer):
        transformer_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
        inps.append(transformer_input)
    return inps


def get_token_len_stats(tokenizer, inputs):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    total_len, max_len = 0, -1
    for inp in inputs:
        inp_len = len(tokenizer(inp).input_ids)
        total_len += inp_len 
        if inp_len > max_len:
            max_len = inp_len
    avg_len = total_len / len(inputs)
    return avg_len, max_len

# Tokenization
def get_transformer_encoding(tokenizer, transformer_inputs, question):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 512, 128

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
    # 0 loss for pad tokens
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels

# Pytorch Dataset
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

# Dataset
def get_dataloader(batch_size, dataset, datatype='train'):
    if type == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)

# %%
class FinetuneTransformer(pl.LightningModule):
    def __init__(self, model_type, model_name, training_dl=None, valid_dl=None, lr=3e-4, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        if model_type == 'T': # for the t5 model
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_type == 'B': # for the bart model
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            print('Unkown Model Type - T or B options only')
        self.training_dataloader = training_dl
        self.valid_dataloader = valid_dl
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
        self.log("validation_loss", loss, on_epoch=True, sync_dist=True)

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
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for training the Transformer Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning Rate for training the Transformer Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=5, help="Total Number of Epochs")
    parser.add_argument("-D", "--num_devices", type=int, default=1, help="Devices used for training")
    parser.add_argument("-MT", "--model_type", type=str, default="t", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", type=str, default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-P", "--prefix_choice", type=int, default=1, help="Choice of prefix used for the input construction - 1, 2, 3")
    params = parser.parse_args()
    return params


# %%
if __name__ == '__main__':
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

    train_inps = construct_transformer_input(train_story, train_answer, args.prefix_choice)
    val_inps = construct_transformer_input(val_story, val_answer, args.prefix_choice)

    # avg_tr_tk_len, max_tr_tk_len = get_token_len_stats(tokenizer, train_inps)
    # avg_val_tk_len, max_val_tk_len = get_token_len_stats(tokenizer, val_inps)

    # with open('token_stats.txt', 'w') as outfile:
    #     print('Average tokenized train length:', avg_tr_tk_len, file=outfile)
    #     print('Max Train Token Length:', max_tr_tk_len, file=outfile)
    #     print('Average tokenized val length:', avg_val_tk_len, file=outfile)
    #     print('Max Val Token Length:', max_tr_tk_len, file=outfile)
    
    if args.model_type == 'T':
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    elif args.model_type == 'B':
        tokenizer = BartTokenizer.from_pretrained(args.model_name)
    else:
        print('Wrong model type - either T or B only')

    train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(tokenizer, train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_transformer_encoding(tokenizer, val_inps, val_question)
    print('Tokenized Data!')

    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')


    batch_size = args.batch_size
    training_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    max_epochs = args.num_epochs
    model = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name, 
        training_dl=training_dataloader, valid_dl=valid_dataloader, 
        num_train_epochs=max_epochs, lr=args.learning_rate)

    # Trainig code
    wandb_logger = WandbLogger(name=args.run_name, project='Quest_Gen_Challenge')

    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    save_directory = os.path.join('./code/finetune/Checkpoints', args.run_name)
    save_checkpoint =  ModelCheckpoint(dirpath=save_directory, monitor='validation_loss', save_top_k=1)

    trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                    default_root_dir=save_directory, 
                    logger=wandb_logger, 
                    max_epochs=max_epochs,
                    callbacks=[early_stop_callback, lr_monitor, save_checkpoint],
                    strategy = DDPStrategy(find_unused_parameters=False))

    trainer.fit(model)

    # if not os.path.exists(save_directory):
    #     os.mkdir(save_directory)
    # model.model.save_pretrained(save_directory)