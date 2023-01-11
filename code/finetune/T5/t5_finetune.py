# %%
'''
python -m code.finetune.T5.t5_finetune \
    -M t5-small \
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
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pdb import set_trace


os.environ['TRANSFORMERS_CACHE'] = '/data/zw16/huggingface/models'
os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneT5'
os.environ['MASTER_PORT'] = '12345'
wandb.login()

# %%
def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# load dataset
def get_parallel_corpus(ip_df, story_df, io_type='story-answer'):
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

        if io_type == 'story-taskAsOutput':
            # The prediction target (stored in the "question" variable) here is: question type + task + question.
            q_type_str = clean_str(row['attribute1'])
            task_str = construct_task(row['answer'], q_type_str)
            question_str = clean_str(row['question'])
            output_str = '[Question attribute] ' + q_type_str + ' [Task] ' + task_str + ' [Question] ' + question_str
            question.append(output_str)
            
        elif io_type == 'story-answer' or io_type == 'story-taskAsInput':
            # The prediction target here is the raw question.
            question.append(clean_str(row['question']))
            question_type.append(clean_str(row['attribute1']))
        else:
            raise Exception('unknown io_type; must be {story-answer, story-taskAsInput, story-taskAsOutput}')

    return story, answer, question, question_type

def get_stats(story, answer, question):
    print('Average story length:', statistics.mean([len(stry) for stry in story]))
    print('Average answer length:', statistics.mean([len(ans) for ans in answer]))
    print('Average question length:', statistics.mean([len(quest) for quest in question]))

# Get the task string according to the question type
def construct_task(answer, question_type):
    if question_type == 'action':
        task = f'Based on the story, write a question that asks about the characters\' behavior or additional information about that behavior. The answer to the question is "{answer}". The question usually starts with "What" or "How."'
    elif question_type == 'setting':
        task = f'Based on the story, write a question that asks about the place or time where/when story events take place. The answer to the question is "{answer}". The question typically starts with "Where" or "When."'
    elif question_type == 'feeling':
        task = f'Based on the story, write a question that asks about the character\'s emotional status or reaction to certain events. The answer to the question is "{answer}". The question is typically worded as "How did/does/do ... feel."'
    elif question_type == 'character':
        task = f'Based on the story, write a question that asks test takers to identify the character of the story, "{answer}," or describe the characteristics of this character. The question usually starts with "Who," "Whose," "What," or "How." The character "{answer}" cannot appear in your question.'
    elif question_type == 'prediction':
        task = f'Based on the story, write a question that asks for a logical consequence. The question usually has the template "What/How/Who will... after/when/if... ?" or "What/How/Who will happen if... ?" The answer to the question is "{answer}". The question cannot contain the phrase "{answer}".'
    elif question_type == 'outcome resolution': # Need some thoughts on this
        task = f'Based on the story, write a question that asks to identify the outcome "{answer}" caused by something in the story. The question is usually worded as "What happened/happens/has happened. . . after...". The question cannot contain the phrase "{answer}".'
    elif question_type == 'causal relationship':
        task = f'Based on the story, write a question that asks about an effect caused by "{answer}". The question usually begins with "Why" or "What made/makes". The question cannot contain the phrase "{answer}".'
    return task

# Constrcut t5 input 
def construct_t5_input(story, answer, question_type, io_type='story-answer'):
    inps = []
    
    if io_type == 'story-answer':
        prefix = 'Generate question from story and answer: '
        for stry, ans in zip(story, answer):
            t5_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
            inps.append(t5_input)
    elif io_type == 'story-taskAsInput':
        for stry, ans, q_type in zip(story, answer, question_type):
            task = construct_task(ans, q_type)
            t5_input = '[Story] ' + stry + ' [Answer] ' + ans + ' [Task] ' + task
            inps.append(t5_input)
    elif io_type == 'story-taskAsOutput':
        prefix = 'Generate question and attribute from story and answer. '
        for stry, ans in zip(story, answer):
            t5_input = prefix + '[Story] ' + stry + ' [Answer] ' + ans
            inps.append(t5_input)

    return inps

def get_token_len_stats(model_name, inputs):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    total_len, max_len = 0, -1
    for inp in inputs:
        inp_len = len(tokenizer(inp).input_ids)
        total_len += inp_len 
        if inp_len > max_len:
            max_len = inp_len
    avg_len = total_len / len(inputs)
    return avg_len, max_len

# Tokenization
def get_t5_encoding(model_name, t5_inputs, question):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
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
    def __init__(self, model_name, niters, lr=3e-4, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # self.training_dataloader = training_dl
        # self.valid_dataloader = valid_dl
        self.hparams.max_epochs = num_train_epochs
        self.hparams.num_train_epochs = num_train_epochs
        self.hparams.warmup_steps = warmup_steps
        self.hparams.lr = lr
        self.hparams.niters = niters
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
            print(self.hparams.niters, file=outfile)
            print(self.hparams.num_train_epochs, file=outfile)
            print(self.hparams.warmup_steps, file=outfile)

        num_train_optimization_steps = self.hparams.num_train_epochs * self.hparams.niters
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(
                                        optimizer,
                                        num_warmup_steps=self.hparams.warmup_steps,
                                        num_training_steps=num_train_optimization_steps
                                        ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    # def train_dataloader(self):
    #     return self.training_dataloader

    # def val_dataloader(self):
    #     return self.valid_dataloader

# %%

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", type=int, default=8, 
                        help="Batch size for training the T5 Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, 
                        help="Learning Rate for training the T5 Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=5, 
                        help="Total Number of Epochs")
    parser.add_argument("-D", "--num_devices", type=int, default=1, 
                        help="Devices used for training")
    parser.add_argument("-M", "--model_name", type=str, default="t5-small", 
                        help="Variant of the T5 model for finetuning")
    parser.add_argument("--io_type", type=str, default='story-answer', 
                        help='Options: story-answer, story-taskAsInput, story-taskAsOutput')
    parser.add_argument("--question_prefix", type=bool, default=False, 
                        help="Whether to use a prefix [Question] before each question. Use only with 'story-taskAsInput'")
    params = parser.parse_args()
    
    return params


# %%
if __name__ == '__main__':
    args = add_params()
    run_name  = f'{args.model_name}.IOType-{args.io_type}.Qprefix-{args.question_prefix}.ep-{args.num_epochs}.bs-{args.batch_size}.lr-{args.learning_rate}.Ndevices-{args.num_devices}'

    story_file = '../../../data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = '../../../data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    val_file = '../../../data/train_val_split_csv/val.csv'
    val_df = pd.read_csv(val_file)

    train_story, train_answer, train_question, train_question_type = get_parallel_corpus(train_df, story_df, args.io_type)
    val_story, val_answer, val_question, val_question_type = get_parallel_corpus(val_df, story_df, args.io_type)
    
    if args.question_prefix:
        for idx in range(len(train_question)):
            train_question[idx] = '[Question] ' + train_question[idx]
        for idx in range(len(val_question)):
            val_question[idx] = '[Question] ' + val_question[idx]
    
    train_inps = construct_t5_input(train_story, train_answer, train_question_type, io_type=args.io_type)
    val_inps = construct_t5_input(val_story, val_answer, val_question_type, io_type=args.io_type)

    # avg_tr_tk_len, max_tr_tk_len = get_token_len_stats(args.model_name, train_inps)
    # avg_val_tk_len, max_val_tk_len = get_token_len_stats(args.model_name, val_inps)

    # with open('token_stats.txt', 'w') as outfile:
    #     print('Average tokenized train length:', avg_tr_tk_len, file=outfile)
    #     print('Max Train Token Length:', max_tr_tk_len, file=outfile)
    #     print('Average tokenized val length:', avg_val_tk_len, file=outfile)
    #     print('Max Val Token Length:', max_tr_tk_len, file=outfile)
    

    train_input_ids, train_attention_mask, train_labels = get_t5_encoding(args.model_name, train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_t5_encoding(args.model_name, val_inps, val_question)
    print('Tokenized Data!')

    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')


    batch_size = args.batch_size
    training_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    max_epochs = args.num_epochs
    model = FinetuneT5(model_name=args.model_name,
                        niters=len(training_dataloader), 
                        num_train_epochs=max_epochs, 
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
    
    save_directory = os.path.join('../Checkpoints', run_name)
    save_checkpoint =  ModelCheckpoint(dirpath=save_directory, monitor='validation_loss', save_top_k=1)

    trainer = Trainer(default_root_dir=save_directory, 
                      logger=wandb_logger, 
                      max_epochs=max_epochs,
                      callbacks=[early_stop_callback, lr_monitor, save_checkpoint],
                    #   strategy="ddp",
                      accelerator='gpu', devices=args.num_devices, 
                      strategy = DeepSpeedStrategy(
                                      stage=3,
                                      offload_optimizer=True,
                                      offload_parameters=True,
                                      )
                      )

    trainer.fit(model, 
                train_dataloaders=training_dataloader, 
                val_dataloaders=valid_dataloader)

    # if not os.path.exists(save_directory):
    #     os.mkdir(save_directory)
    # model.model.save_pretrained(save_directory)
