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

from code.utils.utils import agg_all_metrics, METRICS
from code.score_prediction.models.flan_t5 import ScorePredictionModelFlanT5Wrapper
from code.score_prediction.models.bert import ScorePredictionModelBertWrapper


def add_learner_params():
    parser = argparse.ArgumentParser(description='qg-challenge')

    # Problem definition
    parser.add_argument('--name', default='score_prediction', help='name for the experiment')
    # Model specification
    parser.add_argument('--lm', default='google/flan-t5-small', help='Base language model')
    parser.add_argument('--max_source_length', default=512, type=int, help='Maximum length of input sequence')
    parser.add_argument("--model_folder", type=str, default=None, help="Finetuned model folder relative to saved models dir")
    # Optimizer params for AdamW
    parser.add_argument('--iters', default=10, type=int, help='number of epochs')
    parser.add_argument('--lr', default=3e-4, type=float, help='base learning rate')
    parser.add_argument('--warmup', default=0.1, type=float, help='Fraction of train steps for warmup')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--batch_size_eval', default=64, type=int, help='batch size')
    # Dataloader params
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    parser.add_argument('--data_folder', default="score_prediction", help='Dataset folder name containing train-val-test splits for each cross validation fold')
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

    # Load model wrapper
    if( "flan" in args.lm ):
        model_wrapper = ScorePredictionModelFlanT5Wrapper(args, device)
    elif( "bert" in args.lm ):
        model_wrapper = ScorePredictionModelBertWrapper(args, device)
    else:
        raise "Base LM not supported"
    # Prepare data
    model_wrapper.prepare_data()
    model_wrapper.prepare_dataloaders()
    # Set optimizer and LR scheduler
    model_wrapper.set_optimizer()
    model_wrapper.set_lr_scheduler()
    
    # Multi-GPU training mode
    # TODO p2: multi-gpu training check all code related to multi-gpu training
    if( torch.cuda.device_count() > 1 ):
        model_wrapper.model = nn.DataParallel(model_wrapper.model)

    # Best validation score (loss) to track best model, set to large value
    best_val_score = float('inf')
    # Iteration corresponding to best validation score
    best_iter = -1
    # Test score corresponding to best validation score
    test_score_for_best_val_score = 0

    # Training loop
    with tqdm(range(args.iters)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Epoch {}".format(cur_iter))
            train_loader, val_loader, test_loader = model_wrapper.train_loader, model_wrapper.val_loader, model_wrapper.test_loader
            start_time = time.time()
            
            # Training epoch
            # Set train mode for model
            model_wrapper.set_train_mode()
            train_logs = []
            with tqdm(train_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("Batch {}".format(batch_num))
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logs = model_wrapper.train_step(batch, scaler)  
                    train_logs.append(logs)
            # Push logs to neptune after every training epoch, 
            train_it_time = time.time() - start_time
            train_logs = agg_all_metrics(train_logs)
            if args.neptune:
                run["logs/train/it_time"].log(train_it_time)
                run["metrics/train/loss"].log(train_logs['loss'])
                for metric_name, _ in METRICS:
                    run[f"metrics/train/{metric_name}"].log(train_logs["score_prediction"][metric_name])

            # Evaluation on validation and test sets after every training epoch
            model_wrapper.set_eval_mode()
            test_logs, val_logs, best_val_score, test_score_for_best_val_score, best_iter = evaluate(model_wrapper, best_val_score, test_score_for_best_val_score, best_iter, args, cur_iter, val_loader, test_loader, device, run, saved_models_dir)
            
            # Update training tqdm progress bar
            tepoch.set_postfix({"Train loss" : train_logs['loss'], "Val loss" : val_logs['loss'], "Test loss" : test_logs['loss']})


def evaluate(model_wrapper, best_val_score, test_score_for_best_val_score, best_iter, args, iter, val_loader, test_loader, device, run, saved_models_dir=None):
    # Evaluation on validation and test sets
    test_logs, val_logs = [], []

    # Validation epoch
    val_start_time = time.time()
    with tqdm(val_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model_wrapper.val_step(batch)
            val_logs.append(logs)
    val_it_time = time.time()-val_start_time

    # Test epoch
    test_start_time = time.time()
    with tqdm(test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model_wrapper.test_step(batch)
            test_logs.append(logs)
    test_it_time = time.time()-test_start_time

    # Aggregate logs across batches
    val_logs = agg_all_metrics(val_logs)
    test_logs = agg_all_metrics(test_logs)

    # Update metrics and save best model
    if( float(val_logs["loss"]) < best_val_score ):
        best_val_score = float(val_logs["loss"])
        test_score_for_best_val_score =  float(test_logs["loss"])
        best_iter = iter
        # Save model with best validation score
        dir_best_val_model = os.path.join(saved_models_dir, args.name, run.get_url().split("/")[-1], "best_val_score/")        
        save_model(dir_best_val_model, model_wrapper, iter, float(val_logs["loss"]))

    # Push logs to neptune
    if args.neptune:
        run["metrics/val/best_loss"].log(best_val_score)
        run["metrics/val/loss"].log(val_logs['loss'])
        run["metrics/test/loss"].log(test_logs['loss'])
        run["logs/val/it_time"].log(val_it_time)
        run["logs/test/it_time"].log(test_it_time)
        run["logs/cur_iter"].log(iter)
        run["logs/best_iter"].log(best_iter)
        for dataset, logs in [("val", val_logs), ("test", test_logs)]:
            for metric_name, _ in METRICS:
                run[f"metrics/{dataset}/{metric_name}"].log(logs["score_prediction"][metric_name])
    
    return test_logs, val_logs, best_val_score, test_score_for_best_val_score, best_iter


def unwrap_model(model):
    # Since there could be multiple levels of wrapping (as used in distributed training), unwrap recursively
    # https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/modeling_utils.py
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def save_model(dir_model, model_wrapper, iter, loss):
    model_wrapper.model = unwrap_model(model_wrapper.model)
    pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
    checkpoint = os.path.join(dir_model, "model.pt")
    torch.save({
                'epoch': iter,
                'model_state_dict': model_wrapper.model.state_dict(),
                'optimizer_state_dict': model_wrapper.optimizer.state_dict(),
                'val_loss': loss,
                }, checkpoint)
    # Save tokenizer
    model_wrapper.tokenizer.save_pretrained(dir_model)
    # Save config
    #model.model.config.save_pretrained(dir_model)


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