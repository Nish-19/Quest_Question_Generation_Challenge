"""
Run (finetuned) GPT-2 model for evaluation.

For local validation set:
python -m code.gpt2.evaluate \
    --finetuned \
    --model_folder "gpt2-trial/worldly-star-91" \
    --eval_type "local_val" \
    --eval_folder "train_val_split_csv" \
    --eval_filename "val.csv" \
    --decoding_type "beam_search" \
    --add_instructions \
    --debug 

python -m code.gpt2.evaluate \
    --lm "gpt2" \
    --eval_type "local_val" \
    --eval_folder "train_val_split_csv" \
    --eval_filename "val.csv" \
    --decoding_type "nucleus_sampling_with_top_k" \
    --add_instructions \
    --debug 

For leaderboard public test set:
python -m code.gpt2.evaluate \
    --model_folder "gpt2-trial/worldly-star-91" \
    --eval_type "leaderboard_public_test" \
    --eval_folder "original" \
    --eval_filename "test.csv" 
"""

import os
import pathlib
import json
import argparse
from tqdm import tqdm
import time
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.backends.cudnn as cudnn


from code.utils.create_dataset_split import load_df, save_csv
from code.gpt2.generate import run_gpt2
# create_prompt should be GPT-2 style
from code.gpt2.batch_collator import create_prompt
from code.gpt3.prepare_dataset import load_stories, clean_str, process_multiple_sections


RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--finetuned', action='store_true', help='Whether to use a finetuned GPT-2 model or off-the-shelf from HuggingFace')
    parser.add_argument('--lm', default='gpt2', help='Off-the-shelf GPT-2 model to use if not using a finetuned model', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument("--model_folder", type=str, default="gpt2-trial/worldly-star-91", help="GPT-2 model folder relative to saved models dir")
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    parser.add_argument('--add_instructions', action='store_true', help='Add instructions as a prefix to prompt if not using a finetuned model')
    # GPT-2 generation parameters
    parser.add_argument("--decoding_type", type=str, default="greedy", choices=["greedy", "beam_search", "top_k_sampling", "nucleus_sampling", "nucleus_sampling_with_top_k"], help="Decoding strategy to use")
    #parser.add_argument("--max_tokens", type=int, default=30, help="Maximum number of tokens to generate")
    #parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    #parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of samples to generate")
    #parser.add_argument("--stop", type=str, default='\n', help="Stop sequence")
    # Extras
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seed', default=21, type=int, help='Random seed') 

    params = parser.parse_args()
    
    return params


def save_params(params, name, folder):
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(params.__dict__, f, indent=2)


def evaluate_gpt2(df_eval, story_map, args, model, tokenizer, device):
    # Clean strings
    df_eval["answer"] = df_eval["answer"].apply(clean_str)
    df_eval["source_title"] = df_eval["source_title"].apply(clean_str)

    # Some corresponding sections are a list of sections, we take only the first section
    # TODO: use all sections specified in the list
    df_eval = process_multiple_sections(df_eval)

    # Create prompt 
    df_eval["prompt"] = df_eval.apply(lambda row: create_prompt(row, story_map, args.add_instructions), axis=1)

    # Run GPT-2 model for prompt completion
    tqdm.pandas()
    df_eval["generated_question"] = df_eval.progress_apply(lambda row: run_gpt2(row["prompt"], args, model, tokenizer, device), axis=1)

    return df_eval


def load_model(args, saved_models_dir, device):
    if(args.finetuned):
        # Load finetuned GPT-2 model
        model_folder = os.path.join(saved_models_dir, args.model_folder + "/best_val_loss")
        tokenizer = GPT2Tokenizer.from_pretrained(model_folder)
        model = GPT2LMHeadModel.from_pretrained(model_folder, return_dict=True, pad_token_id=tokenizer.eos_token_id).to(device)
    else:
        # Load off-the-shelf GPT-2 model
        tokenizer = GPT2Tokenizer.from_pretrained(args.lm)
        model = GPT2LMHeadModel.from_pretrained(args.lm, return_dict=True, pad_token_id=tokenizer.eos_token_id).to(device)

    model.eval()

    return model, tokenizer


def main():
    args = add_params()

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

    # Load model and tokenizer
    model, tokenizer = load_model(args, saved_models_dir, device)

    # Load stories
    story_map = load_stories()

    # Load evaluation set
    nrows = 5 if args.debug else None
    folder = os.path.join(RAW_DIR, args.eval_folder)
    df_eval = load_df(args.eval_filename, folder, nrows=nrows)

    # Run GPT-2 model for evaluation
    df_eval = evaluate_gpt2(df_eval, story_map, args, model, tokenizer, device)

    # Save evaluation responses
    folder = os.path.join(RAW_DIR, "results/gpt2/{}".format(args.eval_folder))
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.model_folder.split("/")[-1]
    filename = model_name + "_" + timestr + "_" + args.decoding_type 
    if( args.eval_type == "local_val"):
        save_csv(df_eval, filename, folder)
    else:
        # Save in leaderboard format
        save_csv(df_eval[["pair_id", "generated_question"]], filename, folder)
    
    # Save parameters in a json file for later reference
    save_params(args, filename, folder)


if __name__ == '__main__':
    main()