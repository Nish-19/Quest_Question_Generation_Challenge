"""
With question attribute:

Step 1: Parse each codex run:
python -m code.dataaugmentation.post_process\
    --step 1\
    --attribute\
    --filename "code-davinci-002_20230110-234605_start_0_end_6005_corrected.csv"\
    --run 1

codex_run_0/code-davinci-002_20230109-160155_start_0_end_6005_corrected.csv
codex_run_1/code-davinci-002_20230110-234605_start_0_end_6005_corrected.csv

Step 2: Aggregate all parsed codex runs:
python -m code.dataaugmentation.post_process\
    --step 2\
    --attribute\
    --max_runs 2


Without question attribute:
Run post_process step 1 first then step 2
python -m code.dataaugmentation.post_process --step 1
python -m code.dataaugmentation.post_process --step 2


Manually open parsed files:

- Ensure answer and question columns are not blank
- Ensure count is 8x 6005 = 48040
- Manually search for word "provided" to ensure answer is not "no answer was provided"


Observations on the data augmentation output:

- Some answers don't have a preceeding "The answer is:" (count observed < 20)
Fix: Add "The answer is:", manually inspect these errors though since few of these prompts need to be rerun
- Some outputs have more than 8 QA samples (count observed < 5)
Fix: Keep first 8 QA samples
- Some outputs have less than 8 QA samples (count observed < 5)
Fix: Manually run the prompt again on OpenAI playground with same parameters (codex, temp=0.7, max_length=512, stop=[End])
- Some outputs have missing answers for questions (count observed < 5)
Fix: Manually run the prompt again on OpenAI playground with same parameters (codex, temp=0.7, max_length=512, stop=[End])
- Some outputs have random strings in between QA samples (count observed = 1)
Fix: Manually run the prompt again on OpenAI playground with same parameters (codex, temp=0.7, max_length=512, stop=[End])
- Some answer outputs have "no answer was provided" or empty answers, and usually don't have a preceeding "The answer is:" as well
Fix: Check missing "is:" manually and for ones with "no answer was provided" or empty answers, manually run the prompt again on OpenAI playground with same parameters (codex, temp=0.7, max_length=512, stop=[End])
- Some augmented QA repeat QA seen in prompt, this is okay if target story and prompt story match, not okay otherwise
Fix: Manually run the prompt again on OpenAI playground with same parameters (codex, temp=0.7, max_length=512, stop=[End])
"""

import os
import argparse
import pandas as pd

from code.utils.create_dataset_split import load_df, save_csv
from code.dataaugmentation.create_prompt_data_aug import PROMPT_END_TOKEN, PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE, filter_df
from code.gpt3.prepare_dataset import load_stories
from code.gpt3.evaluate import clean_data


RAW_DIR = "./data"
QUESTION_ATTRIBUTES = ["character", "setting", "action", "feeling", "causal relationship", "outcome resolution", "prediction"]

def add_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--step", type=int, default=2, help="Post processing step to run, run step 1 followed by step 2")
    parser.add_argument("--attribute", action='store_true', help='Question attribute included in augmented data')
    parser.add_argument("--filename", type=str, default="code-davinci-002_20230109-160155_start_0_end_6005_corrected.csv", help="Filename to parse")
    parser.add_argument("--run", type=int, default=0, help="Codex run ID")
    parser.add_argument("--max_runs", type=int, default=1, help="Number of codex runs to aggregate")
    # Params to recreate incontext df used for prompt
    parser.add_argument("--num_qa_examples", type=int, default=8, help="Number of QA examples per story for in-context augmentation prompt")
    parser.add_argument("--num_stories", type=int, default=10, help="Number of stories for in-context augmentation prompt")

    
    params = parser.parse_args()
    
    return params


def split_qa(row, df_incontext, has_attribute):
    x = row["augmented_qa_samples"]
    # Check for errors
    num = 24 if has_attribute else 16
    divisor = 3 if has_attribute else 2
    if( len(x) % divisor != 0 ):
        print(f"\n(Manual) Error: missing A/Q for Q/A in, manually run prompt on OpenAI playground for:\n{row['pair_id']}")
        x = x[:len(x)-(len(x)%divisor)]
    if( len(x) > num ):
        print(f"\nError: More than 8 augmented QA samples for:\n{row['pair_id']}")
        # Keep first 8 QA samples
        x = x[:num]
    elif( len(x) < num ):
        print(f"\n(Manual) Error: Less than 8 augmented QA samples for, manually run prompt on OpenAI playground for:\n{row['pair_id']}")

    l = []
    for sample in x:
        # Check for error
        if( sample == "" ):
            print(f"\n(Manual) Error: Empty string in Q/A, manually run prompt on OpenAI playground for:\n{row['pair_id']}")
        elif("is:" not in sample):
            print(f"\nError: Missing 'is:' in Q/A for:\n{x}")
        else:
            sample = sample.split("is:")[1][1:]
            if( sample == "" ):
                print(f"\n(Manual) Error: Empty string in Q/A, manually run prompt on OpenAI playground for:\n{row['pair_id']}")
        l.append(sample)

    for question, answer in zip(l[1::3], l[2::3]):
        # Check if QA incontext example is repeated in augmented QA
        if( question+answer in (df_incontext["question"] + df_incontext["answer"]).values):
            # Ignore if the target story and prompt story are the same, since it will repeat in this case
            if( row['source_title'] != df_incontext[df_incontext["question"] == question]["source_title"].values[0] ):
                l = "error"
                #print(f"\nError: Prompt QA repeated augmented QA:\n{row['pair_id']}")
                break
    
    return l


def parse_augmented_qa_samples(df, df_incontext, args):
    # Split augmented QA samples
    prompt_end_token = PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE if args.attribute else PROMPT_END_TOKEN
    df["augmented_qa_samples"] = df["augmented_qa_samples"].apply(lambda x: (prompt_end_token + " " + x).split("\n"))
    #df["augmented_qa_samples"] = df["augmented_qa_samples"].apply(lambda x: split_qa(x, args.attribute))
    df["augmented_qa_samples"] = df.apply(lambda row: split_qa(row, df_incontext, args.attribute), axis=1)
    # Separate attributes and questions and answers
    print("No of rows dropped where prompt QA was in repeated augmented QA: ", len(df.loc[df['augmented_qa_samples']=="error"]))
    df.drop(df.loc[df['augmented_qa_samples']=="error"].index, inplace=True)
    if(args.attribute):
        df["attribute1"] = df.apply(lambda row: row["augmented_qa_samples"][0::3], axis=1)
        df["question"] = df.apply(lambda row: row["augmented_qa_samples"][1::3], axis=1)
        df["answer"] = df.apply(lambda row: row["augmented_qa_samples"][2::3], axis=1)
    else:
        df["question"] = df.apply(lambda row: row["augmented_qa_samples"][0::2], axis=1)
        df["answer"] = df.apply(lambda row: row["augmented_qa_samples"][1::2], axis=1)
    # Set other column information to None
    if(args.attribute):
        df.loc[:, ["local_or_sum", "attribute2", "ex_or_im"]] = "None"
    else:
        df.loc[:, ["local_or_sum", "attribute1", "attribute2", "ex_or_im"]] = "None"
    # Explode questions and answers to separate rows
    #orig_len = len(df)
    if(args.attribute):
        df = df.explode(["attribute1", "question", "answer"])
    else:
        df = df.explode(["question", "answer"])
    
    # Drop rows with attribute information not matching accepted question attributes
    print("No of rows dropped where attribute not in accepted question attributes: ", len(df.loc[df['attribute1'].isin(QUESTION_ATTRIBUTES) == False]))
    df.drop(df.loc[df['attribute1'].isin(QUESTION_ATTRIBUTES) == False].index, inplace=True)
    
    # Assert no empty questions or answers
    assert ~df.isnull().values.any(), "Null values found in dataframe"
    assert ~df.eq('').values.any(), "Empty string found in dataframe"
    # Assert 8 QA samples per prompt
    #assert len(df) == orig_len * 8

    return df


def drop_duplicate_samples(df):
        # Count no of duplicate answers
        total_duplicates = len(df[df.duplicated(subset=['answer', 'question'], keep=False)])
        unique_duplicates = df.groupby(['answer', 'question']).size().gt(1).sum()
        print(f"No of duplicate samples removed = {total_duplicates - unique_duplicates} = {round((total_duplicates - unique_duplicates)/len(df) * 100, 2)}%")
        # Remove duplicates where both answer and question are the same
        df = df.drop_duplicates(subset=["answer", "question"], keep="first")

        return df


def main():
    args = add_params()

    if(args.step == 1):
        # Recreate incontext prompt to check if any augmented QA are repeated from incontext QA examples
        folder = os.path.join(RAW_DIR, "train_val_split_csv")
        df_train = load_df("train.csv", folder)
        df_train = clean_data(df_train)
        # Select story with QA examples to use as in-context examples for augmentation prompt
        df_incontext = filter_df(df_train, args)

        # Load unparsed augmented data
        print(f"\n->Parsing file: {args.filename}")
        folder = os.path.join(RAW_DIR, f"augmentation/with_question_attribute/codex_run_{args.run}")
        df = load_df(args.filename, folder)
        df_parsed = parse_augmented_qa_samples(df, df_incontext, args)
        # Save parsed responses
        if(args.attribute):
            df_parsed = df_parsed.drop(columns=["local_or_sum", "attribute2", "ex_or_im", "prompt", "num_words_prompt", "augmented_qa_samples", "num_ex_stories_prompt"])
        else:
            df_parsed = df_parsed.drop(columns=["local_or_sum", "attribute1", "attribute2", "ex_or_im", "prompt", "num_words_prompt", "augmented_qa_samples"])
        
        # Drop duplicate samples
        df_parsed = drop_duplicate_samples(df_parsed)
        save_csv(df_parsed, "parsed", folder)
    
    elif(args.step == 2):
        df_all = pd.DataFrame()
        for run in range(args.max_runs):
            folder = os.path.join(RAW_DIR, f"augmentation/with_question_attribute/codex_run_{run}")
            # Combine all parsed files into a single file
            filename = "parsed.csv"
            df = load_df(filename, folder)
            df_all = pd.concat([df_all, df])
        
        # Drop duplicate samples
        df_all = drop_duplicate_samples(df_all)
        print("No of samples in combined parsed file: ", len(df_all))
        # Print distribution of attributes
        print(df_all["attribute1"].value_counts())
        folder = os.path.join(RAW_DIR, f"augmentation/with_question_attribute/")
        filename = "all_augmented_with_question_attribute_parsed" if args.attribute else "all_augmented_without_question_attribute_parsed"
        save_csv(df_all.reset_index(drop=True), filename, folder)


if __name__ == '__main__':
    main()