# %%
'''
python -m code.finetune.t5_finetune \
    -M t5-small \
    -N t5_small
'''

# %%
import re
import wandb, os
from collections import defaultdict
import argparse
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneT5'

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

# Constrcut t5 input 
def construct_t5_input(story, answer):
    inps = []
    prefix = 'Generate question from story and answer: '
    for stry, ans in zip(story, answer):
        t5_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
        inps.append(t5_input)
    return inps

# Tokenization
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
class FinetuneT5(pl.LightningModule):
    def __init__(self, model_name, lr=3e-4, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
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

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for training the T5 Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning Rate for training the T5 Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=5, help="Total Number of Epochs")
    parser.add_argument("-M", "--model_name", type=str, default="t5-small", help="Variant of the T5 model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
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

    train_inps = construct_t5_input(train_story, train_answer)
    val_inps = construct_t5_input(val_story, val_answer)


    train_input_ids, train_attention_mask, train_labels = get_t5_encoding(train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_t5_encoding(val_inps, val_question)
    print('Tokenized Data!')


    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')


    batch_size = args.batch_size
    train_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    wandb.login()

    max_epochs = args.num_epochs
    model = FinetuneT5(model_name = args.model_name, num_train_epochs=max_epochs, lr=args.learning_rate)

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

    trainer = Trainer(accelerator='gpu', devices=1, 
                    default_root_dir=save_directory, 
                    logger=wandb_logger, 
                    max_epochs=max_epochs,
                    callbacks=[early_stop_callback, lr_monitor, save_checkpoint])

    trainer.fit(model)

    # if not os.path.exists(save_directory):
    #     os.mkdir(save_directory)
    # model.model.save_pretrained(save_directory)