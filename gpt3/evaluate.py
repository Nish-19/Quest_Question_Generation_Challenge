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

In-context learning:
python -m code.gpt3.evaluate \
    --model_name "text-curie-001" \
    --eval_type "local_val" \
    --add_instructions \
    --stop "?" \
    --incontext \
    --incontext_mode "random" \
    --debug

python -m code.gpt3.evaluate \
    --model_name "code-davinci-002" \
    --eval_type "local_val" \
    --add_instructions \
    --stop "?" \
    --incontext \
    --incontext_mode "all_types" \
    --debug
"""

import os
import pathlib
from tqdm import tqdm
import argparse
import json
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from code.utils.create_dataset_split import load_df, save_csv
from code.gpt3.prepare_dataset import load_stories, clean_str, create_prompt
from code.gpt3.run_model import run_gpt3
from code.incontext.create_prompt_incontext import create_prompt_incontext, get_corpus_embedding


RAW_DIR = "./data"
MAX_PARALLEL_PROMPTS_CODEX = 20

def add_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="ada", help="GPT-3 off-the-shelf or finetuned model name")
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--add_instructions', action='store_true', help='Add instructions as a prefix to prompt if not using a finetuned model')
    #parser.add_argument('--batch', action='store_true', help='Batch prompts for generation to avoid rate limit on codex')
    # GPT-3 generation parameters
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--stop", type=str, default='\n', help="Stop sequence")
    # In-context learning parameters
    parser.add_argument('--incontext', action='store_true', help='Use in-context learning')
    parser.add_argument('--incontext_mode', type=str, default="random", choices=["random", "all_types", "retrieval_topk", "retrieval_all_types", "augment"], help="In-context learning mode")
    parser.add_argument("--num_incontext_ex", type=int, default=7, help="Number of incontext examples to use")
    parser.add_argument('--pack_max', action='store_true', help='Pack with max number of in-context examples possible')
    parser.add_argument('--augment_on_single_story', action='store_true', help='Use single story for in-context learning to generate samples from')
    # Retrieval parameters
    parser.add_argument("--embedder_model", type=str, default='all-MiniLM-L6-v2', help="Embedder model for retrieval (mpall-mpnet-base-v2, etc)")
    parser.add_argument("--retrieval_query", type=str, default="story_answer", choices=["story_answer", "story", "answer"], help="Queries to use for retrieval")

    params = parser.parse_args()
    
    return params


def evaluate_gpt3(df_eval, df_train, story_map, args, device):
    tqdm.pandas()
    # Create prompt 
    print(f"Creating prompts...")
    if( args.incontext ):
        if( "retrieval" in args.incontext_mode ):
            embedder = SentenceTransformer(args.embedder_model)
            corpus_embeddings, corpus = get_corpus_embedding(embedder, df_train, story_map, args, device)
            df_eval = df_eval.progress_apply(lambda row: create_prompt_incontext(row, story_map, df_train, df_eval, args, device, embedder, corpus_embeddings, corpus), axis=1)
        else:
            df_eval = df_eval.progress_apply(lambda row: create_prompt_incontext(row, story_map, df_train, df_eval, args, device), axis=1)
    else:
        df_eval["prompt"] = df_eval.progress_apply(lambda row: create_prompt(row, story_map, args.add_instructions), axis=1)

    # Run GPT-3 model for prompt completion
    # Batch prompts to avoid rate limit on Codex
    # https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    print(f"Running model...")
    df_eval["generated_question"] = ""
    for chunk in tqdm(np.split(df_eval, np.arange(MAX_PARALLEL_PROMPTS_CODEX, len(df_eval), MAX_PARALLEL_PROMPTS_CODEX))):
        questions = run_gpt3(chunk["prompt"].tolist(), args)
        chunk["generated_question"] = questions
        df_eval.update(chunk)
    
    # Without batching prompts
    #df_eval["generated_question"] = df_eval.progress_apply(lambda row: run_gpt3(row["prompt"], args), axis=1)

    return df_eval


def save_params(params, name, folder):
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(params.__dict__, f, indent=2)


def clean_data(df):
    # Clean strings
    for col in df.columns:
        df[col] = df[col].astype(str).apply(clean_str)
    
    return df


def main():
    args = add_params()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # For prompting a non finetuned model, i.e., zero shot or few shot learning on off-the-shelf models, add instructions as prefix
    if args.model_name in ["ada", "babbage", "curie", "davinci"]: 
        assert args.add_instructions 
    
    # Load stories
    story_map = load_stories()
    
    # Load evaluation set
    nrows = 50 if args.debug else None
    folder = os.path.join(RAW_DIR, args.eval_folder)
    df_eval = load_df(args.eval_filename, folder, nrows=nrows)
    df_eval = clean_data(df_eval)

    # Load train data for in-context samples
    folder = os.path.join(RAW_DIR, "train_val_split_csv")
    df_train = load_df("train.csv", folder, nrows=nrows)
    df_train = clean_data(df_train)
    # Rename attribute types to one word keywords
    df_train["attribute1"].replace({"causal relationship": "causal", "outcome resolution": "outcome"}, inplace=True)
    
    # Run GPT-3 model for evaluation
    df_eval = evaluate_gpt3(df_eval, df_train, story_map, args, device)
    
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