import torch
import os
import pathlib
import json
import argparse
from tqdm import tqdm
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

from code.utils.create_dataset_split import load_df, save_csv
from code.t5.generate import generate
from code.t5.train import set_random_seed
from code.gpt3.prepare_dataset import load_stories, clean_str
from code.t5.batch_collator import create_prompt


RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument("--model_folder", type=str, default="qg_challenge/QGCHAL-36/cross_val_fold_21/best_val_score/", help="GPT-2 model folder relative to saved models dir")
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="folds/seed_21/train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    # Input specification
    parser.add_argument('--add_instructions', action='store_true', help='Add instructions as a prefix to prompt')
    parser.add_argument('--max_source_length', default=512, type=int, help='Maximum length of input sequence')
    # Generation parameters
    parser.add_argument("--decoding_type", type=str, default="beam_search", choices=["greedy", "beam_search", "contrastive_search", "top_k_sampling", "nucleus_sampling", "nucleus_sampling_with_top_k"], help="Decoding strategy to use")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of questions to generate for each sample")
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seed', default=21, type=int, help='Random seed') 

    params = parser.parse_args()
    
    return params


def save_params(params, name, folder):
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(params.__dict__, f, indent=2)


def evaluate(df_eval, story_map, args, model, tokenizer, device):
    # Clean strings
    df_eval["answer"] = df_eval["answer"].apply(clean_str)
    df_eval["source_title"] = df_eval["source_title"].apply(clean_str)
    # Create prompt 
    df_eval["prompt"] = df_eval.apply(lambda row: create_prompt(row, story_map, args.add_instructions), axis=1)
    # Run model for generation
    tqdm.pandas()
    df_eval["generated_question"] = df_eval.progress_apply(lambda row: generate(row["prompt"], args, model, tokenizer, device), axis=1)
    # Explode generated_question column since num_return_sequences >= 1
    df_eval = df_eval.explode("generated_question")
    
    return df_eval


def load_model(args, saved_models_dir, device):
    model_folder = os.path.join(saved_models_dir, args.model_folder)
    tokenizer = T5Tokenizer.from_pretrained(model_folder)
    model = T5ForConditionalGeneration.from_pretrained(model_folder, return_dict=True).to(device)
    model.eval()

    return model, tokenizer


def save_generated_questions(df_eval, args):
    # Save evaluation responses
    folder = os.path.join(RAW_DIR, "results/flan_t5", args.eval_folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.model_folder.split("/")[1]
    filename = model_name + "_" + timestr + "_" + args.decoding_type 
    if( args.eval_type == "local_val"):
        save_csv(df_eval, filename, folder)
    else:
        # Save in leaderboard format
        save_csv(df_eval[["pair_id", "generated_question"]], filename, folder)
    # Save generation parameters in a json file for reference
    save_params(args, filename, folder)


def main():
    args = add_params()
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

    # Load model and tokenizer
    model, tokenizer = load_model(args, saved_models_dir, device)
    # Load stories
    story_map = load_stories()
    # Load evaluation set
    nrows = 5 if args.debug else None
    folder = os.path.join(RAW_DIR, args.eval_folder)
    df_eval = load_df(args.eval_filename, folder, nrows=nrows)

    # Run question generation model for evaluation
    df_eval = evaluate(df_eval, story_map, args, model, tokenizer, device)

    # Save generated questions
    save_generated_questions(df_eval, args)


if __name__ == '__main__':
    main()