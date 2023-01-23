import os
import pandas as pd

from code.utils.create_dataset_split import SEED
from code.gpt3.prepare_dataset import get_story
from code.incontext.create_prompt_incontext import get_length


RAW_DIR = "./data/"
number_to_cardinal_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
number_to_ordinal_word = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh"}
SEP_TOKEN = "\n"
PROMPT_END_TOKEN = f'The {number_to_ordinal_word[1]} question is:'
PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE = f'The {number_to_ordinal_word[1]} attribute is:'
PROMPT_START_TOKEN = f"Each question attribute is explained below."
MAX_WORDS = 6000
# Attributes sorted in ascending order of frequency of occurence in the dataset
ATTRIBUTES_SORTED = ["prediction", "setting", "feeling", "outcome resolution", "character", "causal relationship", "action"]


def filter_df_balance_attributes(df_train, args):
    # Group by source_title and cor_section, 
    df_train["count"] = df_train.groupby(["source_title", "cor_section"])["attribute1"].transform(len)
    # Count frequency of each question attribute1 in each group
    for attr in ATTRIBUTES_SORTED:
        df_train[attr.split(" ")[0] + "_count"] = df_train.groupby(["source_title", "cor_section"])["attribute1"].transform(lambda x: x.value_counts().get(attr, default=0))
    df_filter = df_train.drop_duplicates(["source_title", "cor_section"])
    # Filter groups which have QA example count >= args.num_qa_examples
    df_filter = df_filter[df_filter['count'] >= args.num_qa_examples]
    # Sort groups by descending order of attribute frequency, scanning from least to most frequent attributes
    df_filter = df_filter.sort_values(by=["prediction_count", "setting_count", "feeling_count", "outcome_count", "character_count", "causal_count", "action_count"], ascending=False)

    # Add args.num_qa_examples QA examples for each story
    df_train['key'] = df_train['source_title'] + df_train['cor_section']
    # Keep track of stories already added to df_incontext, so we don't repeat stories
    current_unique_stories = []
    df_incontext = pd.DataFrame()
    for source_title, cor_section in zip(df_filter['source_title'], df_filter['cor_section']):
        df_story_sec = df_train[df_train['key'].isin([source_title + cor_section])].drop(['key'], axis=1)
        # If the story with the same or diferent section is not already added to df_incontext, and it has at least 5 unique attributes
        if( ( source_title not in current_unique_stories ) and ( len(df_story_sec["attribute1"].unique()) >= 5 ) ):
            current_unique_stories.append(source_title)
            num_ex = 0
            break_while = False
            while(True):
                # Add QA examples starting from least to most frequent attributes
                for attribute in ATTRIBUTES_SORTED:
                    df_ex = df_story_sec[df_story_sec['attribute1'] == attribute].head(1)
                    if( len(df_ex) != 0 ):
                        df_incontext = pd.concat([df_incontext, df_ex])
                        # Remove the added example from df_story_sec by index
                        df_story_sec = df_story_sec.drop(df_ex.index)
                        num_ex += 1
                        if( num_ex == args.num_qa_examples ):
                            break_while = True
                            break
                if( break_while ):
                    break
        if( len(current_unique_stories) == args.num_stories ):
            break
        
    return df_incontext


def filter_df(df_train, args):
    # Group by source_title and cor_section, 
    df_group = df_train.groupby(['source_title', 'cor_section']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
    
    # Filter groups which have QA example count >= args.num_qa_examples
    df_filter = df_group[df_group['count'] >= args.num_qa_examples]
    # Remove duplicate stories, and keep the first occurence (since we have sorted by count) with most QA examples
    df_filter = df_filter.drop_duplicates(subset=['source_title'])
    print(f"No of unique stories with >={args.num_qa_examples} QA examples: {len(df_filter)}")
    print(df_filter.head(10))

    # Keep args.num_stories random stories
    df_filter = df_filter.sample(n=args.num_stories, random_state=SEED)

    # Filter df_train to only include samples matching story and section from df_filter
    df_train['key'] = df_train['source_title'] + df_train['cor_section']
    df_train = df_train[df_train['key'].isin(df_filter['source_title'] + df_filter['cor_section'])].drop(['key'], axis=1)
    assert len(df_train["source_title"].unique()) == args.num_stories

    # Randomly select args.num_qa_examples QA examples from each story
    df_incontext = df_train.groupby(['source_title']).apply(lambda x: x.sample(n=args.num_qa_examples, random_state=SEED)).reset_index(drop=True)
    print(df_incontext)

    return df_incontext

    
def get_attributes_desc():
    filepath = os.path.join(RAW_DIR, "augmentation/attributes_desc.txt")
    with open(filepath, "r") as f:
        attributes_desc = f.read()
    
    return attributes_desc


def create_prompt_data_aug(row, df_incontext, story_map, args):
    prompt = ""
    target_story = get_story(row, story_map)

    if( args.question_attributes_desc ):
        attributes_desc = get_attributes_desc()
        prompt += f"{PROMPT_START_TOKEN}{SEP_TOKEN}{attributes_desc}{SEP_TOKEN}"

    # Add story and QA examples to prompt
    num_ex_stories = 0
    for story in df_incontext["source_title"].unique():
        curr_ex_text = ""
        df_incontext_story = df_incontext[df_incontext["source_title"] == story]
        instructions = f"Write {number_to_cardinal_word[args.num_qa_examples]} questions and answers for the {number_to_ordinal_word[num_ex_stories+1]} story."
        # Add current story to prompt
        curr_ex_text += f"[Begin]{SEP_TOKEN}{instructions}{SEP_TOKEN}The {number_to_ordinal_word[num_ex_stories+1]} story is: {get_story(df_incontext_story.iloc[0], story_map)}{SEP_TOKEN}"
        # Add QA examples for current story to prompt
        i = 0
        for index, incontext_row in df_incontext_story.iterrows():
            if( args.question_attribute_after ):
                curr_ex_text += f'The {number_to_ordinal_word[i+1]} question is: {incontext_row["question"]}{SEP_TOKEN}The {number_to_ordinal_word[i+1]} answer is: {incontext_row["answer"]}{SEP_TOKEN}The {number_to_ordinal_word[i+1]} attribute is: {incontext_row["attribute1"]}{SEP_TOKEN}'
            elif( args.question_attribute_before ):
                curr_ex_text += f'The {number_to_ordinal_word[i+1]} attribute is: {incontext_row["attribute1"]}{SEP_TOKEN}The {number_to_ordinal_word[i+1]} question is: {incontext_row["question"]}{SEP_TOKEN}The {number_to_ordinal_word[i+1]} answer is: {incontext_row["answer"]}{SEP_TOKEN}'
            else:
                curr_ex_text += f'The {number_to_ordinal_word[i+1]} question is: {incontext_row["question"]}{SEP_TOKEN}The {number_to_ordinal_word[i+1]} answer is: {incontext_row["answer"]}{SEP_TOKEN}'
            i += 1
        curr_ex_text += f"[End]{SEP_TOKEN}"
        # 384 = 3/4 * 512 tokens reserved for completion
        if( (get_length(prompt + curr_ex_text + target_story) + 384) > MAX_WORDS ):
            break
        prompt += curr_ex_text
        num_ex_stories += 1
    
    # Add target story for augmentation
    prompt_end_token = PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE if args.question_attribute_before else PROMPT_END_TOKEN
    instructions = f"Write {number_to_cardinal_word[args.num_qa_examples]} questions and answers for the {number_to_ordinal_word[num_ex_stories+1]} story."
    prompt += f"[Begin]{SEP_TOKEN}{instructions}{SEP_TOKEN}The {number_to_ordinal_word[num_ex_stories+1]} story is: {target_story}{SEP_TOKEN}{prompt_end_token}"
    
    row["prompt"] = prompt
    row["num_ex_stories_prompt"] = num_ex_stories
    row["num_words_prompt"] = get_length(prompt)

    return row


def explore_prompt(df):
    print(f'No of unique stories: {len(df["source_title"].unique())}')
    print(f'No of unique examples: {len(df)}')
    print(f'No of unique attributes: {len(df["attribute1"].unique())}')
    # Explore question attribute1 tags distribution
    print(df['attribute1'].value_counts(normalize=True))