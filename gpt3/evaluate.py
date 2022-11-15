"""
Run (finetuned) GPT-3 model for evaluation.

For local validation set:
python -m code.gpt3.evaluate \
    --model_name "curie:ft-umass-amherst:curie-train-2022-11-03-00-04-39" \
    --eval_type "local_val" \
    --debug 

For leaderboard public test set:
python -m code.gpt3.evaluate \
    --model_name "curie:ft-umass-amherst:curie-train-2022-11-03-00-04-39" \
    --eval_type "leaderboard_public_test" \
    --eval_folder "original" \
    --eval_filename "test.csv" 
"""

import os
import pathlib
from tqdm import tqdm
import argparse
import json
import time

from code.utils.create_dataset_split import load_df, save_csv
from code.gpt3.prepare_dataset import load_stories, clean_str, create_prompt, process_multiple_sections
from code.gpt3.run_model import run_gpt3


RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="ada", help="GPT-3 off-the-shelf or finetuned model name")
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--add_instructions', action='store_true', help='Add instructions as a prefix to prompt if not using a finetuned model')
    # GPT-3 generation parameters
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--stop", type=str, default='\n', help="Stop sequence")

    params = parser.parse_args()
    
    return params


def evaluate_gpt3(df_eval, story_map, args):
    # Clean strings
    df_eval["answer"] = df_eval["answer"].apply(clean_str)
    df_eval["source_title"] = df_eval["source_title"].apply(clean_str)

    # Some corresponding sections are a list of sections, we take only the first section
    # TODO: use all sections specified in the list
    df_eval = process_multiple_sections(df_eval)

    # Create prompt 
    df_eval["prompt"] = df_eval.apply(lambda row: create_prompt(row, story_map, args.add_instructions), axis=1)

    # Run GPT-3 model for prompt completion
    tqdm.pandas()
    df_eval["generated_question"] = df_eval.progress_apply(lambda row: run_gpt3(row["prompt"], args), axis=1)

    return df_eval


def save_params(params, name, folder):
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(params.__dict__, f, indent=2)


def main():
    args = add_params()
    
    # For prompting a non finetuned model, i.e., zero shot or few shot learning on off-the-shelf models, add instructions as prefix
    if args.model_name in ["ada", "babbage", "curie", "davinci"]: 
        assert args.add_instructions 
    
    # Load stories
    story_map = load_stories()
    
    # Load evaluation set
    nrows = 5 if args.debug else None
    folder = os.path.join(RAW_DIR, args.eval_folder)
    df_eval = load_df(args.eval_filename, folder, nrows=nrows)
    
    # Run GPT-3 model for evaluation
    df_eval = evaluate_gpt3(df_eval, story_map, args)
    
    # Save evaluation responses
    folder = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.model_name.replace(":", "_")
    filename = model_name + "_" + timestr
    if( args.eval_type == "local_val"):
        save_csv(df_eval, filename, folder)
    else:
        # Save in leaderboard format
        save_csv(df_eval[["pair_id", "generated_question"]], filename, folder)
    
    # Save parameters in a json file for later reference
    save_params(args, filename, folder)


if __name__ == '__main__':
    main()