import os

from code.utils.create_dataset_split import SEED
from code.gpt3.prepare_dataset import get_story
from code.incontext.create_prompt_incontext import get_length


RAW_DIR = "./data/"
number_to_cardinal_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
number_to_ordinal_word = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
SEP_TOKEN = "\n"
PROMPT_END_TOKEN = f'The {number_to_ordinal_word[1]} question is:'
PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE = f'The {number_to_ordinal_word[1]} attribute is:'
PROMPT_START_TOKEN = f"Each question attribute is explained below."
MAX_WORDS = 6000


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
    df_incontext = df_train[df_train['key'].isin(df_filter['source_title'] + df_filter['cor_section'])].drop(['key'], axis=1)
    #print(len(df_incontext["attribute1"].unique()))
    assert len(df_incontext["source_title"].unique()) == args.num_stories

    # Randomly select args.num_qa_examples QA examples from each story
    df_incontext = df_incontext.groupby(['source_title']).apply(lambda x: x.sample(n=args.num_qa_examples, random_state=SEED)).reset_index(drop=True)
    #print(len(df_incontext))
    #print(len(df_incontext["attribute1"].unique()))
    print(df_incontext)

    return df_incontext

    
def get_attributes_desc():
    filepath = os.path.join(RAW_DIR, "augmentation/attributes_desc.txt")
    with open(filepath, "r") as f:
        attributes_desc = f.read()
    
    return attributes_desc


def create_prompt_data_aug(row, df_incontext, story_map, args):
    prompt = ""
    instructions = f"Write {number_to_cardinal_word[args.num_qa_examples]} questions and answers for the story."
    target_story = get_story(row, story_map)

    if( args.question_attributes_desc ):
        attributes_desc = get_attributes_desc()
        prompt += f"{PROMPT_START_TOKEN}{SEP_TOKEN}{attributes_desc}{SEP_TOKEN}"

    # Add story and QA examples to prompt
    num_ex_stories = 0
    for story in df_incontext["source_title"].unique():
        curr_ex_text = ""
        df_incontext_story = df_incontext[df_incontext["source_title"] == story]
        # Add current story to prompt
        curr_ex_text += f"[Begin]{SEP_TOKEN}{instructions}{SEP_TOKEN}The story is: {get_story(df_incontext_story.iloc[0], story_map)}{SEP_TOKEN}"
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
        if( get_length(prompt + curr_ex_text + target_story) > MAX_WORDS ):
            break
        prompt += curr_ex_text
        num_ex_stories += 1
    
    # Add target story for augmentation
    prompt_end_token = PROMPT_END_TOKEN_WITH_ATTRIBUTE_BEFORE if args.question_attribute_before else PROMPT_END_TOKEN
    prompt += f"[Begin]{SEP_TOKEN}{instructions}{SEP_TOKEN}The story is: {target_story}{SEP_TOKEN}{prompt_end_token}"
    
    row["prompt"] = prompt
    row["num_ex_stories_prompt"] = num_ex_stories
    row["num_words_prompt"] = get_length(prompt)

    return row


def explore_prompt(df_data_aug):
    # Explore question attribute1 tags distribution
    print(df_data_aug['attribute1'].value_counts(normalize=True))