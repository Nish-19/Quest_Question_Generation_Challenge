"""
Prepare training set for finetuning GPT-3.
IMPORTANT: Then run CLI data preparation tool on prepared training set: openai tools fine_tunes.prepare_data -f <LOCAL_FILE> 

To finetune:
openai api fine_tunes.create \
    -t ./data/gpt3_finetuning/train.jsonl \
    -m curie \
    --n_epochs 4 \
    --suffix curie_train
"""

import os
import collections
import re
import pathlib

from code.utils.create_dataset_split import load_df


RAW_DIR = "./data/"
#PROMPT_END_TOKEN = "\n\n###\n\n"
PROMPT_END_TOKEN = "The question is:"
SEP_TOKEN = "\n"
COMPLETION_END_TOKEN = "\n"
INSTRUCTIONS = "Write a question based on the story and answer."


def get_story(row, story_map):
    sections = str(row["cor_section"]).split(",")
    story = " ".join([story_map[row["source_title"]][int(section)] for section in sections])

    return story


def verify_length(df_train, max_words=1536):
    # Verify length of prompt + completion <= 2048 tokens on 0.75*2048=1536 words
    df_train["prompt_len"] = df_train["prompt"].apply(lambda x: len(x.split()))
    df_train["completion_len"] = df_train["completion"].apply(lambda x: len(x.split()))
    df_train["total_len"] = df_train["prompt_len"] + df_train["completion_len"]
    print(df_train[["prompt_len", "completion_len", "total_len"]].describe())
    assert df_train["total_len"].max() <= max_words


def prepare_dataset(df_train, story_map):
    # Clean strings
    df_train["answer"] = df_train["answer"].apply(clean_str)
    df_train["question"] = df_train["question"].apply(clean_str)
    df_train["source_title"] = df_train["source_title"].apply(clean_str)
    
    # Check if answer and question contains the SEP_TOKEN or PROMPT_END_TOKEN or COMPLETION_END_TOKEN
    for tok in [SEP_TOKEN, PROMPT_END_TOKEN, COMPLETION_END_TOKEN]:
        assert df_train["answer"].str.contains(tok).sum() == 0
        assert df_train["question"].str.contains(tok).sum() == 0
    
    # Create prompt and corresponding completion
    df_train["prompt"] = df_train.apply(lambda row: create_prompt(row, story_map), axis=1)
    df_train["completion"] = df_train.apply(lambda row: create_completion(row), axis=1)
    #print(df_train[["prompt", "completion"]].head())

    # Verify length of prompt + completion <= 2048 tokens on 0.75*2048=1536 words
    verify_length(df_train)

    return df_train[["prompt", "completion"]]


def create_prompt(row, story_map, add_instructions=False):
    story = get_story(row, story_map)
    # Suffix prompt with PROMPT_END_TOKEN
    # Separate story and answer with SEP_TOKEN
    if( add_instructions ):
        # For prompting a non finetuned model, i.e., zero shot or few shot learning on off-the-shelf models, add instructions as prefix
        prompt = f"""{INSTRUCTIONS}{SEP_TOKEN}The story is: {story}{SEP_TOKEN}The answer is: {row['answer']}{SEP_TOKEN}{PROMPT_END_TOKEN}"""
    else:
        # If using a finetuned model, no instructions are needed as prefix
        prompt = f"""The story is: {story}{SEP_TOKEN}The answer is: {row['answer']}{SEP_TOKEN}{PROMPT_END_TOKEN}"""

    return prompt


def create_completion(row):
    # Precede completion text with a single space
    # Suffix completetion with END_TOKEN
    completion = f""" {row['question']}{COMPLETION_END_TOKEN}"""

    return completion


def clean_str(text):
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_story_map(df_stories, assertion=True):
    df_stories["source_title"] = df_stories["source_title"].apply(clean_str)
    story_map = collections.defaultdict(dict)
    for index, row in df_stories.iterrows():
        # Clean strings since stories have \n in between sentences
        story_text = clean_str(row["text"])
        # Check if story contains the SEP_TOKEN or PROMPT_END_TOKEN or COMPLETION_END_TOKEN
        if( assertion ):
            assert SEP_TOKEN not in story_text and PROMPT_END_TOKEN not in story_text and COMPLETION_END_TOKEN not in story_text
        story_map[row["source_title"]][row["cor_section"]] = story_text
    
    #print("Num stories:", len(story_map))
    #print("Story map:", {k: story_map[k] for k in list(story_map)[:2]})
    
    return story_map


def save_dataset(df, filename, folder):
    filename = os.path.join(folder, filename)
    # JSONL file with display double quotes escaped with a backslash which is OK
    # The example for Case study: Entity extraction also has an escaped double quote (https://beta.openai.com/docs/guides/fine-tuning/case-study-entity-extraction)
    df.to_json(filename + ".jsonl", orient="records", lines=True)
    df[:10].to_json(filename + "_small.jsonl", orient="records", lines=True)


def load_stories(folder=None, assertion=True):
    folder = os.path.join(RAW_DIR, "original") if folder is None else folder
    df_stories = load_df("source_texts.csv", folder)
    story_map = create_story_map(df_stories, assertion)

    return story_map


def main():
    # Load stories
    story_map = load_stories()
    # Load train data
    folder = os.path.join(RAW_DIR, "train_val_split_csv")
    df_train = load_df("train.csv", folder)
    
    # Prepare dataset for GPT-3 finetuning
    df_train = prepare_dataset(df_train, story_map)

    # Save dataset in jsonl format
    folder = os.path.join(RAW_DIR, "gpt3_finetuning")
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    save_dataset(df_train, "train", folder)
    

if __name__ == '__main__':
    main()