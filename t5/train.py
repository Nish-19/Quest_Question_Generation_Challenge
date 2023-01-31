import torch
from torch import nn
import torch.backends.cudnn as cudnn
import random
import argparse
import time
import pathlib
import os
from tqdm import tqdm
import numpy as np
import neptune.new as neptune

from code.utils.utils import agg_all_metrics
from code.t5.model import QuestionGenerationModelWrapper


def add_learner_params():
    parser = argparse.ArgumentParser(description='qg-challenge')

    # Problem definition
    parser.add_argument('--name', default='qg_challenge', help='name for the experiment')
    # Model specification
    parser.add_argument('--lm', default='google/flan-t5-small', help='Base language model')
    parser.add_argument('--add_instructions', action='store_true', help='Add instruction prefix to the input')
    parser.add_argument('--max_source_length', default=512, type=int, help='Maximum length of input sequence')
    parser.add_argument('--max_target_length', default=128, type=int, help='Maximum length of output sequence')
    # Model checkpoint params if continuing training
    parser.add_argument('--checkpoint', action='store_true', help='Continue training from model checkpoint')
    parser.add_argument('--model_folder', default=None, help='Model folder relative to saved models dir')
    # Optimizer params for AdamW
    parser.add_argument('--iters', default=5, type=int, help='number of epochs')
    parser.add_argument('--lr', default=3e-4, type=float, help='base learning rate')
    parser.add_argument('--warmup', default=0.1, type=float, help='Fraction of train steps for warmup')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--batch_size_eval', default=64, type=int, help='batch size')
    # Dataloader params
    parser.add_argument('--augment', action='store_true', help='Augment on external data')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    parser.add_argument('--data_folder', default="folds", help='Dataset folder name containing train-val-test splits for each cross validation fold')
    parser.add_argument('--data_augment_folder', default="augmentation", help='Dataset folder name containing train-val-test splits for each cross validation fold')
    parser.add_argument('--cross_val_fold', default=21, type=int, help='Cross validation fold to use')
    # Misc
    parser.add_argument('--neptune_project', default="ni9elf/qg-challenge", help='name of neptune project')
    parser.add_argument('--seed', default=21, type=int, help='random seed')  
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--neptune', action='store_true', help='link training to neptune')
    parser.add_argument('--debug', action='store_true', help='debug mode with less data')
    parser.add_argument('--torch_debug', action='store_true', help='Torch autograd debug mode which slows down computation for finding NaNs')
    # Automatic mixed precision training -> faster training without affecting accuracy on Volta (V100) or Turing (RTX8000) GPUs
    parser.add_argument('--amp', action='store_true', help='apply automatic mixed precision training')

    params = parser.parse_args()
    
    return params


def train(args, run, device, saved_models_dir):
    if( args.amp ):
        # Using pytorch automatic mixed precision (fp16/fp32) for faster training
        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Prepare model
    model = QuestionGenerationModelWrapper(args, device, saved_models_dir)

    # Multi-GPU training mode
    # TODO: multi-gpu training check all code related to multi-gpu training
    if( torch.cuda.device_count() > 1 ):
        model.model = nn.DataParallel(model.model)

    # Best validation score (loss) to track best model, set to large value
    best_val_score = float('inf')
    # Iteration corresponding to best validation score
    best_iter = -1

    # Training loop
    with tqdm(range(args.iters)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Epoch {}".format(cur_iter))
            train_loader, val_loader = model.train_loader, model.val_loader
            start_time = time.time()

            # Training epoch
            # Set train mode for model
            model.set_train_mode()
            train_logs = []
            with tqdm(train_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("Batch {}".format(batch_num))
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logs = model.train_step(batch, scaler)  
                    train_logs.append(logs)
            # Push logs to neptune after every training epoch, 
            train_it_time = time.time() - start_time
            train_logs = agg_all_metrics(train_logs)
            if args.neptune:
                run["logs/train/it_time"].log(train_it_time)
                run["metrics/train/loss"].log(train_logs['loss'])

            # Evaluation on validation and test sets after every training epoch
            model.set_eval_mode()
            val_logs, best_val_score, best_iter = evaluate(model, best_val_score, best_iter, args, cur_iter, val_loader, device, run, saved_models_dir)
            # Update training tqdm progress bar
            tepoch.set_postfix({"Train loss" : train_logs['loss'], "Val loss" : val_logs['loss']})


def evaluate(model, best_val_score, best_iter, args, iter, val_loader, device, run, saved_models_dir=None):
    # Evaluation on validation and test sets
    val_logs = []

    # Validation epoch
    val_start_time = time.time()
    with tqdm(val_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model.val_step(batch)
            val_logs.append(logs)
    val_it_time = time.time()-val_start_time

    # Aggregate logs across batches
    val_logs = agg_all_metrics(val_logs)
    
    # Update metrics and save best model
    if( float(val_logs["loss"]) < best_val_score ):
        best_val_score = float(val_logs["loss"])
        best_iter = iter
        # Save model with best validation score
        dir_best_val_model = os.path.join(saved_models_dir, args.name, run.get_url().split("/")[-1], "cross_val_fold_{}".format(args.cross_val_fold), "best_val_score/")
        save_model(dir_best_val_model, model)

    # Push logs to neptune
    if args.neptune:
        run["metrics/val/loss"].log(val_logs['loss'])
        run["metrics/val/best_loss"].log(best_val_score)
        run["logs/val/it_time"].log(val_it_time)
        run["logs/cur_iter"].log(iter)
        run["logs/best_iter"].log(best_iter)
    
    return val_logs, best_val_score, best_iter


def save_model(dir_model, model):
    pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
    # Save model
    if isinstance(model.model, torch.nn.DataParallel):
        model.model.module.model.save_pretrained(dir_model)
    else:
        model.model.model.save_pretrained(dir_model)
    # Save tokenizer
    model.model.tokenizer.save_pretrained(dir_model)


def set_random_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True


def main():
    args = add_learner_params()
    # Set torch debug mode if applicable
    torch.autograd.set_detect_anomaly(args.torch_debug)
    # Set saved models dir
    if ( torch.cuda.is_available() ):
        saved_models_dir = "/work/nigel_umass_edu/qg_challenge/saved_models/"
    else:
        saved_models_dir = "../saved_models/"
    # Set random seed if specified
    if args.seed != -1:
        set_random_seed(args.seed)
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'
    # Link training to neptune
    run = None
    if args.neptune:
        run = neptune.init_run(project = args.neptune_project, name = args.name)  
        run["parameters"] = vars(args)
    # Train model
    train(args, run, device, saved_models_dir)


if __name__ == '__main__':
    main()