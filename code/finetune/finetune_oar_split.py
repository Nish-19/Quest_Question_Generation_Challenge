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
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneTransformer'


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
        suffix = ''
    elif choice == 2:
        prefix = 'Generate question: '
        suffix = ''
    elif choice == 3:
        prefix = ''
        suffix = ''
    elif choice == 4:
        prefix = 'Generate question from story and answer: '
        suffix = '\nThe question is:'
    for stry, ans in zip(story, answer):
        transformer_input = prefix + '\nThe story is ' + stry + '\nThe answer is ' + ans + suffix
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
    def __init__(self, model_type, model_name, lp=False, training_dl=None, valid_dl=None, lr=3e-4, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        if model_type == 'T': # for the t5 model
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_type == 'B': # for the bart model
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            print('Unkown Model Type - T or B options only')
        # Check Linear Probing
        if lp:
            for name, param in self.model.named_parameters():
                if 'DenseReluDense' in name or 'layer_norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.model.shared.requires_grad = True
            self.model.lm_head.requires_grad = True
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
    parser.add_argument('-W', '--wandb', action=argparse.BooleanOptionalAction, help='For Wandb logging')
    parser.add_argument('-TFN', '--train_file_name', type=str, default="train.csv", help="Training File name")
    parser.add_argument('-TS', '--training_strategy', type=str, default="DP", help="DP for dataparalle and DS for deepspeed")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for training the Transformer Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning Rate for training the Transformer Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=5, help="Total Number of Epochs")
    parser.add_argument("-D", "--num_devices", type=int, default=1, help="Devices used for training")
    parser.add_argument('-LP', '--linear_probing', action=argparse.BooleanOptionalAction, help='For Linear Probing (Train only the lm head)')
    parser.add_argument("-MT", "--model_type", type=str, default="t", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", type=str, default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
    parser.add_argument('-LC', '--load_checkpoint', action=argparse.BooleanOptionalAction, help='Load Checkpoint for re-finetuning')
    parser.add_argument("-CN", "--checkpoint_name", type=str, default="flan_t5_large_codex_0.00_augment", help="Variant of the trained Transformer Base Model")
    parser.add_argument("-P", "--prefix_choice", type=int, default=1, help="Choice of prefix used for the input construction - 1, 2, 3")
    params = parser.parse_args()
    return params


# %%
if __name__ == '__main__':
    args = add_params()

    story_file = './data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)


    # NOTE: Load Train-Val Folds split here
    for fold in range(4, 5):
        print('Training Fold {:d}'.format(fold))
        train_file = os.path.join('./data/score_prediction/qg_model/fold_{:d}/train_val_split_csv/'.format(fold), args.train_file_name)
        train_df = pd.read_csv(train_file)
        val_file = './data/score_prediction/qg_model/fold_{:d}/train_val_split_csv/val.csv'.format(fold)
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

        # NOTE: Load checkpoint
        if args.load_checkpoint:
            search_dir = os.path.join('./code/finetune/Checkpoints_new', args.checkpoint_name)
            for file in os.listdir(search_dir):
                ckpt_file = os.path.join(search_dir, file)
            print('ckpt_file', ckpt_file)
            # model_pl = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name)
            model = FinetuneTransformer.load_from_checkpoint(ckpt_file, model_type = args.model_type)
            print('Successfully loaded the saved checkpoint!')
            save_name = 'reft_' + args.run_name
        else:
            model = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name, 
                lp=args.linear_probing, training_dl=training_dataloader, 
                valid_dl=valid_dataloader, num_train_epochs=max_epochs, 
                lr=args.learning_rate)
            if args.linear_probing:
                save_name = 'lp_' + args.run_name
            else:
                save_name = args.run_name


        save_path = os.path.join('./code/finetune/Checkpoints_new', save_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        save_name = save_name + '_' + str(fold)
        print('Save name:', save_name)

        # Trainig code
        if args.wandb:
            wandb.login()
            logger = WandbLogger(name=save_name, project='Quest_Gen_Challenge')
        else:
            logger = CSVLogger("run_results", name=save_name)


        early_stop_callback = EarlyStopping(
            monitor='validation_loss',
            patience=3,
            strict=False,
            verbose=False,
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        save_directory = os.path.join(save_path, save_name)
        save_checkpoint =  ModelCheckpoint(dirpath=save_directory, monitor='validation_loss', save_top_k=1)

        if args.training_strategy == 'DP':
            strategy = DDPStrategy(find_unused_parameters=False)
        elif args.training_strategy == 'DS':
            strategy = DeepSpeedStrategy(stage=3,
                                        offload_optimizer=True,
                                        offload_parameters=True)


        trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                        default_root_dir=save_directory, 
                        logger=logger, 
                        max_epochs=max_epochs,
                        callbacks=[early_stop_callback, lr_monitor, save_checkpoint],
                        strategy = strategy)

        trainer.fit(model)