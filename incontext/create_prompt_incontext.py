from random import Random
from code.gpt3.prepare_dataset import INSTRUCTIONS, SEP_TOKEN, get_story


SEED = 21
random = Random(SEED)


def get_length(s):
    return len(s.split())


def create_prompt_incontext(row, story_map, df_train, args):
    target_story = get_story(row, story_map)
    # Max tokens for code-davinci-002 is 4000 while for text-curie-001 is 2000
    # Max  words <= 4/3 * max_tokens
    max_words = 3000 if "code" in args.model_name else 1500
    samples = get_incontext_samples(row, story_map, df_train, args)
    samples_text = ""
    num = 0
    for sample in samples:
        if(args.incontext_mode == "all_types"):
            # TODO: add attribute type info to incontext example text and try to predict the attribute type
            curr_text = f"[Begin]{SEP_TOKEN}The story is: {sample[0]}{SEP_TOKEN}The answer is: {sample[1]}{SEP_TOKEN}The question is: {sample[2]}{SEP_TOKEN}[End]{SEP_TOKEN}"
        else:
            curr_text = f"[Begin]{SEP_TOKEN}The story is: {sample[0]}{SEP_TOKEN}The answer is: {sample[1]}{SEP_TOKEN}The question is: {sample[2]}{SEP_TOKEN}[End]{SEP_TOKEN}"
        if( get_length(samples_text + curr_text + target_story) > max_words ):
            break
        samples_text += curr_text
        num += 1

    # Suffix prompt with PROMPT_END_TOKEN
    prompt_end_token = "The question is:" if args.incontext_mode != "all_types" else "The question is:"
    # Separate story and answer with SEP_TOKEN
    if( args.add_instructions ):
        # For prompting a non finetuned model, i.e., zero shot or few shot learning on off-the-shelf models, add instructions as prefix
        prompt = f"{samples_text}[Begin]{SEP_TOKEN}The story is: {target_story}{SEP_TOKEN}The answer is: {row['answer']}{SEP_TOKEN}{prompt_end_token}"
    else:
        # If using a finetuned model, no instructions are needed as prefix
        prompt = f"{samples_text}[Begin]{SEP_TOKEN}The story is: {target_story}{SEP_TOKEN}The answer is: {row['answer']}{SEP_TOKEN}{prompt_end_token}"

    row["prompt"] = prompt
    row["num_words_prompt"] = get_length(prompt)
    row["num_incontext_ex"] = num
    
    return row


def get_incontext_samples_random(row, story_map, df_train, args):
    max_num = 100 if args.pack_max else args.num_incontext_ex
    # Get random samples from the training set
    sample_rows = df_train.sample(max_num)
    samples = []
    for _, sample_row in sample_rows.iterrows():
        samples.append((get_story(sample_row, story_map), sample_row["answer"], sample_row["question"]))
    
    return samples


def get_incontext_samples_all_types(row, story_map, df_train, args):
    # Get list of all types
    attributes = df_train["attribute1"].unique()

    # If pack_max then get many samples of each attribute else only one from each attribute
    max_num_per_type = 10 if args.pack_max else 1
    # Get random samples for each type without replacement
    samples = []
    for _ in range(max_num_per_type):
        samples_per_attribute = []
        for attribute in attributes:
            sample_row = df_train[(df_train["attribute1"] == attribute)].sample(1)
            samples_per_attribute.append((get_story(sample_row.iloc[0], story_map), sample_row.iloc[0]["answer"], sample_row.iloc[0]["question"], attribute))
            # Sample without replacement
            df_train = df_train.drop(sample_row.index)
        # Shuffle per attribute samples
        random.shuffle(samples_per_attribute)
        samples += samples_per_attribute
    
    return samples


def get_incontext_samples_retrieval(row, story_map, df_train):
    pass


def get_incontext_samples(row, story_map, df_train, args):
    # Remove current row from df_train based on column pair_id
    df_train = df_train[df_train["pair_id"] != row["pair_id"]]

    # Rename attribute types to one word keywords
    df_train["attribute1"].replace({"causal relationship": "causal", "outcome resolution": "outcome"}, inplace=True)

    if( args.incontext_mode == "random" ):
        samples = get_incontext_samples_random(row, story_map, df_train, args)
    elif( args.incontext_mode == "all_types" ):
        samples =  get_incontext_samples_all_types(row, story_map, df_train, args)
    elif( args.incontext_mode == "retrieval" ):
        samples =  get_incontext_samples_retrieval(row, story_map, df_train, args)
    else:
        raise ValueError(f"Unknown get incontext_samples mode: {args.incontext_mode}")
    
    return samples