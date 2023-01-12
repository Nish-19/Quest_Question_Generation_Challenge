"""
Augment training set using Codex.

python3.9 -m code.dataaugmentation.augment \
    --model_name "code-davinci-002" \
    --stop "[End]" \
    --start_row 0 \
    --end_row 1 \
    --question_attribute_before \
    --question_attributes_desc
"""
import os
import argparse
import pathlib
import time
from tqdm import tqdm
import numpy as np

from code.utils.create_dataset_split import load_df, save_csv
from code.gpt3.evaluate import clean_data, save_params
from code.gpt3.prepare_dataset import load_stories
from code.gpt3.run_model import run_gpt3
from code.dataaugmentation.create_prompt_data_aug import filter_df, create_prompt_data_aug, explore_prompt, PROMPT_END_TOKEN, SEP_TOKEN


RAW_DIR = "./data"
MAX_PARALLEL_PROMPTS_CODEX = 1


def add_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="code-davinci-002", help="GPT-3 off-the-shelf or finetuned model name")
    parser.add_argument('--debug', action='store_true', help='Debug mode for augmenting on a small subset of 5 samples')
    # Incontext augmenration parameters
    parser.add_argument("--num_stories", type=int, default=10, help="Number of stories for in-context augmentation prompt")
    parser.add_argument("--start_row", type=int, default=0, help="Number of stories for in-context augmentation prompt")
    parser.add_argument("--end_row", type=int, default=6005, help="Number of stories for in-context augmentation prompt")
    # NUM_QA_EXAMPLES = 5 gives train_set like distribution over question attribute tags
    # NUM_QA_EXAMPLES = 8 is also an acceptable distribution with prediction tag being 4.3%
    # NUM_QA_EXAMPLES = 10 excludes prediction tag
    parser.add_argument("--num_qa_examples", type=int, default=8, help="Number of QA examples per story for in-context augmentation prompt")
    parser.add_argument('--question_attribute_before', action='store_true', help='Add question attribute before each incontext QA example in prompt')
    parser.add_argument('--question_attribute_after', action='store_true', help='Add question attribute after each incontext QA example in prompt')
    parser.add_argument('--question_attributes_desc', action='store_true', help='Add explanation of all question attributes as a prefix header to the prompt')
    # Codex generation parameters
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--best_of", type=int, default=1, help="Number of samples to take the best n, best_of must be >= n")
    parser.add_argument("--stop", type=str, default='[End]', help="Stop sequence")

    params = parser.parse_args()
    
    return params


def get_openai_api_keys():
    api_keys = []
    for i in range(0, 5):
        api_key = os.getenv(f"OPENAI_API_KEY_{i}")
        if api_key:
            api_keys.append(api_key)
    
    return api_keys


def augment_data(df_train, df_incontext, story_map, args):
    tqdm.pandas()
    # Create prompt 
    print(f"Creating prompts...")
    df_train = df_train.progress_apply(lambda row: create_prompt_data_aug(row, df_incontext, story_map, args), axis=1)
    # Augment using Codex
    api_keys = get_openai_api_keys()
    print(f"Running model for augmentation...")
    df_train["augmented_qa_samples"] = ""
    for chunk in tqdm(np.split(df_train, np.arange(MAX_PARALLEL_PROMPTS_CODEX, len(df_train), MAX_PARALLEL_PROMPTS_CODEX))):
        out_text = run_gpt3(chunk["prompt"].tolist(), args, api_keys)
        chunk["augmented_qa_samples"] = out_text
        df_train.update(chunk)
    # Explode augmented_qa_samples column since top-n augmentation completion (each completion has k QA samples) candidates could have been generated
    df_train = df_train.explode("augmented_qa_samples")
    # Parse augmented_qa_samples column
    #df_train = parse_augmented_qa_samples(df_train)

    return df_train


def main():
    args = add_params()

    # Load stories
    story_map = load_stories()

    # Load train data to augment
    folder = os.path.join(RAW_DIR, "train_val_split_csv")
    df_train = load_df("train.csv", folder)
    df_train = clean_data(df_train)

    # Select story with QA examples to use as in-context examples for augmentation prompt
    df_incontext = filter_df(df_train, args)
    #explore_prompt(df_incontext)

    # Run Codex model for augmentation
    args.start_row = 0 if args.debug else args.start_row
    args.end_row = 5 if args.debug else args.end_row
    df_augment = augment_data(df_train[args.start_row:args.end_row], df_incontext, story_map, args)

    # Save evaluation responses
    folder = os.path.join(RAW_DIR, "augmentation")
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.model_name.replace(":", "_")
    filename = model_name + "_" + timestr + "_start_" + str(args.start_row) + "_end_" + str(args.end_row) 
    df_augment = df_augment.drop(columns=["key"])#, "augmented_qa_samples"])
    save_csv(df_augment, filename, folder)

    # Save parameters in a json file for later reference
    save_params(args, filename, folder)


if __name__ == '__main__':
    main()