"""
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
"""

import os
import argparse
import pandas as pd

from code.utils.create_dataset_split import load_df, save_csv
from code.dataaugmentation.create_prompt_data_aug import PROMPT_END_TOKEN


RAW_DIR = "./data"


def split_qa(x):
    # Check for errors
    if( len(x) % 2 != 0 ):
        print(f"\n(Manual) Error: missing A/Q for Q/A in, manually run prompt on OpenAI playground for:\n{x}")
        x = x[:len(x)-1]
    if( len(x) > 16 ):
        print(f"\nError: More than 8 augmented QA samples for:\n{x}")
        # Keep first 8 QA samples
        x = x[:16]
    elif( len(x) < 16 ):
        print(f"\n(Manual) Error: Less than 8 augmented QA samples for, manually run prompt on OpenAI playground for:\n{x}")

    l = []
    for sample in x:
        # Check for error
        if( sample == "" ):
            print(f"\n(Manual) Error: Empty string in Q/A, manually run prompt on OpenAI playground for:\n{x}")
        elif("is:" not in sample):
            print(f"\nError: Missing 'is:' in Q/A for:\n{x}")
        else:
            sample = sample.split("is:")[1][1:]
            if( sample == "" ):
                print(f"\n(Manual) Error: Empty string in Q/A, manually run prompt on OpenAI playground for:\n{x}")
        l.append(sample)
    
    return l


def parse_augmented_qa_samples(df):
    # Split augmented QA samples
    df["augmented_qa_samples"] = df["augmented_qa_samples"].apply(lambda x: (PROMPT_END_TOKEN + " " + x).split("\n"))
    df["augmented_qa_samples"] = df["augmented_qa_samples"].apply(lambda x: split_qa(x))
    # Separate questions and answers
    df["answer"] = df.apply(lambda row: row["augmented_qa_samples"][1::2], axis=1)
    df["question"] = df.apply(lambda row: row["augmented_qa_samples"][0::2], axis=1)
    # If attribute information is not available, set to None
    df.loc[:, ["local_or_sum", "attribute1", "attribute2", "ex_or_im"]] = "None"
    # Explode questions and answers to separate rows
    orig_len = len(df)
    df = df.explode(["answer", "question"])
    # Assert no empty questions or answers
    #assert ~df.isnull().values.any(), "Null values found in dataframe"
    #assert ~df.eq('').values.any(), "Empty string found in dataframe"
    # Assert 8 QA samples per prompt
    #assert len(df) == orig_len * 8

    return df


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2, help="Post processing step to run, run step 1 followed by step 2")
    params = parser.parse_args()
    
    return params


def main():
    args = add_params()

    filenames = ["code-davinci-002_20230108-202236_start_989_end_992.csv",
                ]

    if(args.step == 1):
        # Load unparsed augmented data
        for filename in filenames:
            print(f"\n->Parsing file: {filename}")
            folder = os.path.join(RAW_DIR, "augmentation")
            df = load_df(filename, folder)
            df_parsed = parse_augmented_qa_samples(df)
            # Save parsed responses
            df_parsed = df_parsed.drop(columns=["local_or_sum", "attribute1", "attribute2", "ex_or_im", "prompt", "num_words_prompt", "augmented_qa_samples"])
            save_csv(df_parsed, filename.split(".")[0]+"_parsed", folder)
    
    elif(args.step == 2):
        # Combine all parsed files into a single file
        df_all = pd.DataFrame()
        for filename in filenames:
            filename = filename.split(".")[0]+"_parsed.csv"
            folder = os.path.join(RAW_DIR, "augmentation")
            df = load_df(filename, folder)
            df_all = pd.concat([df_all, df])
        
        # Count no of duplicate answers
        total_duplicates = len(df_all[df_all.duplicated(subset=['answer', 'question'], keep=False)])
        unique_duplicates = df_all.groupby(['answer', 'question']).size().gt(1).sum()
        print(f"No of duplicate answers removed = {total_duplicates - unique_duplicates} = {round((total_duplicates - unique_duplicates)/len(df_all) * 100, 2)}%")
        # Remove duplicates where both answer and question are the same
        df_all = df_all.drop_duplicates(subset=["answer", "question"], keep="first")
        
        # Save combined file
        save_csv(df_all.reset_index(drop=True), "all_"+filenames[0].split("_start")[0]+"_parsed", folder)


if __name__ == '__main__':
    main()