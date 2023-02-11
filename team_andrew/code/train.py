import os
import json
import argparse
import pandas as pd
from transformers import T5Tokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from code.fairy_dataset import FairyDataset
from code.model import LanguageModel
from code.utils import get_transformer_encoding, get_dataloader, load_df, get_parallel_corpus, construct_transformer_input


def add_params():
    parser = argparse.ArgumentParser()
    # Optimizer params for AdamW
    parser.add_argument("-B", "--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-E", "--num_epochs", type=int, default=1, help="Total number of epochs")
    parser.add_argument("-W", "--warmup_steps", type=int, default=1000, help="Total number of epochs")
    parser.add_argument('-LP', '--linear_probing', action=argparse.BooleanOptionalAction, help='For linear probing (train only the lm head)')
    # Model specification
    parser.add_argument("-MN", "--model_name", type=str, default="google/flan-t5-xl", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="devout-jazz", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-LEN", "--max_source_length", type=int, default=512, help="Max input sequence length to for tokenizer")
    # Lightning trainer params
    parser.add_argument("-ACC", "--accumulate_grad_batches", type=int, default=6, help="Num of batches to accumulate gradients for")
    parser.add_argument("-PRE", "--precision", type=int, default=32, help="Precision for training")
    parser.add_argument("-CLIP", "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("-D", "--num_devices", type=int, default=1, help="Devices used for training")
    # Misc
    parser.add_argument('-DG', '--debug', action=argparse.BooleanOptionalAction, help='For Debugging')
    params = parser.parse_args()
    
    return params


if __name__ == '__main__':
    # Seed
    seed_everything(21, workers=True)

    # Load settings 
    args = add_params()
    with open('./SETTINGS.json', 'r') as infile:
        settings = json.load(infile)

    # Load data
    data_dir = settings['RAW_DATA_DIR']
    nrows = 32 if args.debug else None
    train_df = load_df('train.csv', data_dir, nrows=nrows)
    story_df = load_df('source_texts.csv', data_dir)
    # Match story, question and answer
    train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
    # Prepare prompts for language model
    train_inps = construct_transformer_input(train_story, train_answer)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    # Tokenize data
    train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(tokenizer, train_inps, train_question, max_source_length=args.max_source_length, max_target_length=64)
    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    # Prepare dataloader
    training_dataloader = get_dataloader(args.batch_size, train_dataset, datatype='train')

    # Prepare model
    model = LanguageModel(args, training_dataloader)
        
    # Prepare lightning trainer
    logger = CSVLogger("run_results", name=args.run_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    save_directory = os.path.join(settings['MODEL_DIR'], args.run_name)
    save_checkpoint =  ModelCheckpoint(dirpath=save_directory, save_last=True)
    # Training strategy
    strategy = DeepSpeedStrategy(stage = 2,
                                offload_optimizer=True,
                                allgather_bucket_size=5e8,
                                reduce_bucket_size=5e8
                                )
    trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                    default_root_dir=save_directory, 
                    logger=logger,
                    max_epochs=args.num_epochs,
                    callbacks=[lr_monitor, save_checkpoint],
                    deterministic=True,
                    strategy = strategy,
                    accumulate_grad_batches=args.accumulate_grad_batches,
                    gradient_clip_val=args.gradient_clip_val,
                    precision=args.precision)
    # Train model
    trainer.fit(model)
    print('Model training complete')
    print('Saving model in path: {:s}'.format(save_directory))