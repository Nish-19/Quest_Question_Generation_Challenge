import sys
import os
import re
import json
import wandb
from collections import defaultdict
import argparse
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.optimization import Adafactor
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger, CSVLogger, NeptuneLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from deepspeed.ops.adam import DeepSpeedCPUAdam

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneTransformer'
seed_everything(21, workers=True)

# Tokenization
def get_transformer_encoding(tokenizer, transformer_inputs, question):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 512, 64

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
    if datatype == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)

# %%
class FinetuneTransformer(pl.LightningModule):
    def __init__(self, model_name, lp=False, training_dl=None, valid_dl=None, lr=3e-4, num_train_epochs=5, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
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
        #optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.lr)
        #optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)

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

    
def add_params():
    parser = argparse.ArgumentParser()
    # NOTE: redundant for now
    parser.add_argument('-RT', "--run_type", type=str, default="finetune_local", choices=["finetune_local", "finetune_all", "finetune_local_with_aug", "finetune_all_with_aug"], help="Train-val split type for run")
    #parser.add_argument('-NEP', '--neptune', action=argparse.BooleanOptionalAction, help='For Neptune logging')
    parser.add_argument('-TS', '--training_strategy', type=str, default="DP", help="DP for dataparalle and DS for deepspeed")
    parser.add_argument("-B", "--batch_size", type=int, default=3, help="Batch size for training the Transformer Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning Rate for training the Transformer Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=8, help="Total Number of Epochs")
    parser.add_argument("-D", "--num_devices", type=int, default=1, help="Devices used for training")
    parser.add_argument('-LP', '--linear_probing', action=argparse.BooleanOptionalAction, help='For Linear Probing (Train only the lm head)')
    parser.add_argument("-MN", "--model_name", type=str, default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
    parser.add_argument('-LC', '--load_checkpoint', action=argparse.BooleanOptionalAction, help='Load Checkpoint for re-finetuning')
    parser.add_argument("-CN", "--checkpoint_name", type=str, default="flan_t5_large_codex_0.00_augment", help="Variant of the trained Transformer Base Model")
    parser.add_argument("-ACC", "--accumulate_grad_batches", type=int, default=6, help="Num of batches to accumulate gradients for")
    parser.add_argument("-PRE", "--precision", type=int, default=32, help="Precision for training")
    parser.add_argument("-CLIP", "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument('-DG', '--debug', action=argparse.BooleanOptionalAction, help='For Debugging')
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    args = add_params()

    # Open settings file
    with open('SETTINGS.json', 'r') as infile:
        json_file = json.load(infile)

    # Load data
    train_path = json_file['TRAIN_DATA_CLEAN_PATH']
    if args.debug:
        train_df = pd.read_csv(train_path)[:32]
    else:
        train_df = pd.read_csv(train_path)


    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    print('Loaded T5 tokenizer!')
    
    train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(tokenizer, train_df['Transformer Input'].tolist(), train_df['Transformer Output'].tolist())
    print('Tokenized Data!')

    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    print('Created Pytorch Dataset')


    batch_size = args.batch_size
    training_dataloader = get_dataloader(batch_size, train_dataset)
    print('Loaded Dataloader!')

    max_epochs = args.num_epochs

    # NOTE: Load checkpoint
    if args.load_checkpoint:
        search_dir = os.path.join('./Checkpoints', args.checkpoint_name)
        for file in os.listdir(search_dir):
            ckpt_file = os.path.join(search_dir, file)
        print('ckpt_file', ckpt_file)
        model = FinetuneTransformer.load_from_checkpoint(ckpt_file)
        print('Successfully loaded the saved checkpoint!')
        save_name = 'reft_' + args.run_name
    else:
        model = FinetuneTransformer(model_name = args.model_name, 
            lp=args.linear_probing, training_dl=training_dataloader, 
            num_train_epochs=max_epochs, lr=args.learning_rate)
        
        save_name = args.run_name

    if args.linear_probing:
        save_name = 'lp_' + save_name
            
    print('Save name:', save_name)

    logger = CSVLogger("run_results", name=save_name)


    lr_monitor = LearningRateMonitor(logging_interval='step')
        
    save_directory = os.path.join('./Checkpoints', save_name)
    save_checkpoint =  ModelCheckpoint(dirpath=save_directory, save_last=True)

    if args.training_strategy == 'DP':
        strategy = DDPStrategy(find_unused_parameters=False)
    elif args.training_strategy == 'DS':
        strategy = DeepSpeedStrategy(stage = 2,
                                    offload_optimizer=True,
                                    allgather_bucket_size=5e8,
                                    reduce_bucket_size=5e8
                                    )


    trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                    default_root_dir=save_directory, 
                    logger=logger,
                    max_epochs=max_epochs,
                    callbacks=[lr_monitor, save_checkpoint],
                    deterministic=True,
                    strategy = strategy,
                    accumulate_grad_batches=args.accumulate_grad_batches,
                    gradient_clip_val=args.gradient_clip_val,
                    precision=args.precision)

    trainer.fit(model)

    print('Model Training Complete!')
    print('Saving model in path: {:s}'.format(save_directory))