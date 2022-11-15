"""
Training script for GPT-2 training.

python3.9 -m code.gpt2.train \
    --name "gpt2-trial" \
    --lm gpt2 \
    --data_folder "train_val_split_json" \
    --debug --iters 1 \
    --batch_size 4 \
    --lm_loss_location "question" \
    --log_wandb 


python3.9 -m code.gpt2.train \
    --name "gpt2-trial" \
    --lm gpt2 \
    --data_folder "train_val_split_json" \
    --debug --iters 1 \
    --batch_size 4 \
    --lm_loss_location "all" \
    --log_wandb 
"""

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import argparse
import random
import pathlib
import os
import time
from tqdm import tqdm
import wandb

from code.utils.utils import agg_all_metrics
from code.gpt2.model import LanguageModelBaseWrapper


# disable warnings in hugging face logger
#logging.set_verbosity_error()


def add_learner_params():
    parser = argparse.ArgumentParser(description='qg_challenge')

    parser.add_argument('--name', default='train', help='Name for the experiment')
    parser.add_argument('--wandb_project', default="qg-challenge", help='Name of weights and biases project')
    # Optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const')
    parser.add_argument('--opt', default='adam', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=100, type=int, help='Number of epochs')
    parser.add_argument('--warmup', default=0, type=float, help='Number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=2e-5, type=float, help='Base learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    # GPT-2 params
    parser.add_argument('--lm', default='gpt2', help='GPT-2 model to use', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--lm_loss_location', default='adam', help='On what part of the input to compute CLM loss on', choices=['all', 'question', 'question_answer'])
    parser.add_argument('--add_instructions', action='store_false', help='Add instructions prefix before prompt to GPT-2')
    # Trainer params
    parser.add_argument('--eval_freq', default=1, type=int, help='Epoch frequency for evaluation')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    parser.add_argument('--seed', default=21, type=int, help='Random seed') 
    # Extras
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--log_wandb', action='store_true', help='Log training to weights and biases')
    parser.add_argument('--debug', action='store_true', help='Debug mode with less data')
    # Automatic mixed precision training -> faster training but might affect accuracy slightly
    parser.add_argument('--amp', action='store_true', help='Apply automatic mixed precision training')
    # Data loading
    parser.add_argument('--data_folder', default="train_val_split_json", help='Dataset folder name containing train-val-test splits for each cross validation fold')

    params = parser.parse_args()
    
    return params


def train(args, device, saved_models_dir):
    if( args.amp ):
        # Using pytorch automatic mixed precision (fp16/fp32) for faster training
        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Prepare data and model for training
    model = LanguageModelBaseWrapper(args, device)
    model.prepare_data()

    # Multi-GPU training mode
    if( torch.cuda.device_count() > 1 ):
        model.model = nn.DataParallel(model.model)

    # Training variables
    cur_iter = 0
    # Best validation metric
    best_val_metric = float("inf")

    # Training loop
    with tqdm(range(args.iters)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Epoch {}".format(cur_iter+1))
            train_loader, val_loader = model.dataloaders()
            start_time = time.time()
            cur_iter += 1
            
            # Train epoch
            # Set train mode for model
            model.set_train_mode()
            train_logs = []
            with tqdm(train_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("Batch {}".format(batch_num))
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logs = model.train_step(batch, scaler)  
                    train_logs.append(logs)
            
            # After every training epoch, push logs to weights and biases
            train_it_time = time.time() - start_time
            train_logs = agg_all_metrics(train_logs)
            if args.log_wandb:
                wandb.log({"logs/train/it_time": train_it_time})
                wandb.log({"metrics/train/loss": train_logs['loss']})
                wandb.log({"logs/train/cur_iter" : cur_iter})
            
            # Evaluate on validation set after every training epoch
            val_logs, best_val_metric = evaluate(model, best_val_metric, args, val_loader, device, saved_models_dir=saved_models_dir)
            
            # Update training tqdm progress bar
            tepoch.set_postfix({"Train loss" : train_logs['loss'], "Val loss" : val_logs['loss']})


def evaluate(model, best_val_metric, args, val_loader, device, saved_models_dir=None):
    # Evaluation epoch
    # Set eval mode for model
    model.set_eval_mode()
    val_logs = []
    eval_start_time = time.time()
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logs = model.eval_step(batch)
        val_logs.append(logs)
    eval_it_time = time.time()-eval_start_time
    
    # Aggregate logs across batches
    val_logs = agg_all_metrics(val_logs)
    # Update metrics and save model
    if( float(val_logs["loss"]) < best_val_metric ):
        best_val_metric = float(val_logs["loss"])
        # Save model with best validation CLM loss
        dir_best_valid_metric = saved_models_dir + args.name + "/" + wandb.run.name + "/" + "/best_val_loss/"
        save_model(dir_best_valid_metric, model)
    # Push logs to weights and biases
    if args.log_wandb:
        wandb.log({"metrics/val/loss": val_logs['loss']})
        wandb.log({"metrics/val/best_loss": best_val_metric})
        wandb.log({"logs/val/it_time": eval_it_time})
    
    return val_logs, best_val_metric
         

def save_model(dir_model, model):
    # TODO: check we can recover a model
    pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
    # Save tokenizer
    model.tokenizer.save_pretrained(dir_model)
    # Save model
    if isinstance(model.model, torch.nn.DataParallel):
        model.model.module.model.save_pretrained(dir_model)
    else:
        model.model.model.save_pretrained(dir_model)
    #filename_model = os.path.join(dir_model, "model_state_dict.pt")
    #torch.save({"model_state_dict" : model.model.state_dict()}, filename_model)


def main():
    args = add_learner_params()

    if ( torch.cuda.is_available() ):
        # Unity server saved models dir
        saved_models_dir = "/work/nigel_umass_edu/qg_challenge/saved_models/"
    else:
        # Local saved models dir
        saved_models_dir = "../saved_models/"
    
    # Set random seed if specified
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'
    
    # Link training to weights and biases
    if args.log_wandb:
        wandb.init(project=args.wandb_project, entity="ni9elf")
        wandb.config.update(vars(args))

    # Train model
    train(args, device, saved_models_dir)


if __name__ == '__main__':
    main()