from random import Random
from sentence_transformers.util import semantic_search

from code.gpt3.prepare_dataset import INSTRUCTIONS, SEP_TOKEN, get_story


SEED = 21
random = Random(SEED)


def get_length(s):
    return len(s.split())


def create_prompt_incontext(row, story_map, df_train, df_eval, args, device, embedder=None, corpus_embeddings=None, corpus=None):
    target_story = get_story(row, story_map)
    # Max tokens for code-davinci-002 is 4000 while for text-curie-001 is 2000
    # Max  words <= 4/3 * max_tokens
    max_words = 3000 if "code" in args.model_name else 1500
    samples, retrieval_query = get_incontext_samples(row, story_map, df_train, df_eval, args, device, embedder, corpus_embeddings, corpus)
    samples_text = ""
    num = 0
    if( args.augment_on_single_story ):
        samples_text += f"The story is: {target_story}{SEP_TOKEN}"
        for sample in samples:
            curr_text = f"The answer is: {sample[1]}{SEP_TOKEN}The question is: {sample[2]}{SEP_TOKEN}"
            if( get_length(samples_text + curr_text) > max_words ):
                break
            samples_text += curr_text
            num += 1
    else:
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
    if( args.augment_on_single_story ):
        prompt = f"{samples_text}The answer is: {row['answer']}{SEP_TOKEN}{prompt_end_token}"
    else:
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
    row["retrieval_query"] = retrieval_query

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
    attributes = get_attribute_types(df_train)

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


def get_attribute_types(df_train):
    # Get list of all attribute types
    return df_train["attribute1"].unique()


def get_corpus_embedding(embedder, df_train, story_map, args, device):
    df_train["retrieval_corpus"] = df_train.apply(lambda row: get_retrieval_query(row, story_map, args), axis=1)
    if( args.incontext_mode == "retrieval_topk" ):
        corpus = df_train["retrieval_corpus"].tolist()
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True) 
        corpus_embeddings = corpus_embeddings.to(device)
    elif( args.incontext_mode == "retrieval_all_types" ):
        corpus = {}
        corpus_embeddings = {}
        attributes = get_attribute_types(df_train)
        for attribute in attributes:
            corpus[attribute] = df_train[(df_train["attribute1"] == attribute)].reset_index(drop=True)
            corpus_embeddings[attribute] = embedder.encode(corpus[attribute]["retrieval_corpus"].tolist(), convert_to_tensor=True) 
            corpus_embeddings[attribute] = corpus_embeddings[attribute].to(device)
    else:
        raise ValueError(f"Invalid incontext mode: {args.incontext_mode}")

    return corpus_embeddings, corpus


def get_retrieval_query(row, story_map, args):
    if( args.retrieval_query == "story_answer" ):
        query = f"The story is: {get_story(row, story_map)}{SEP_TOKEN}The answer is: {row['answer']}"
    elif( args.retrieval_query == "story" ):
        query = f"The story is: {get_story(row, story_map)}"
    elif( args.retrieval_query == "answer" ):
        query = f"The answer is: {row['answer']}"
    else:
        raise ValueError(f"Invalid retrieval query {args.retrieval_query}")
    
    return query


def get_incontext_samples_retrieval_topk(row, story_map, df_train, args, device, embedder, corpus_embeddings):
    # Get query
    query = get_retrieval_query(row, story_map, args)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.to(device)
    # Semantic search for top k samples
    retrieved_out = semantic_search(query_embedding, corpus_embeddings, top_k=args.num_incontext_ex)
    samples = []
    for entry in retrieved_out[0]:
        sample_row = df_train.iloc[entry["corpus_id"]]
        samples.append((get_story(sample_row, story_map), sample_row["answer"], sample_row["question"]))
    # Return samples in reverse order which corresponds to lowest to highest similarity score since LMs focus on recent words
    samples = samples[::-1]

    return samples, query


def get_incontext_samples_retrieval_all_types(row, story_map, df_train, args, device, embedder, corpus_embeddings, corpus):
    samples = []
    # Get query
    query = get_retrieval_query(row, story_map, args)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.to(device)
    # Get list of all attribute types
    attributes = get_attribute_types(df_train)
    # Semantic search for top 1 sample per question attribute type
    for attribute in attributes:
        retrieved_out = semantic_search(query_embedding, corpus_embeddings[attribute], top_k=1)
        sample_row = corpus[attribute].iloc[retrieved_out[0][0]["corpus_id"]]
        samples.append((get_story(sample_row, story_map), sample_row["answer"], sample_row["question"], attribute))
    # Shuffle top-1 samples across types
    random.shuffle(samples)

    return samples, query


def get_augmented_samples(row, story_map, df_eval, args):
    # Remove target row from df_eval using pair_id
    df_eval = df_eval[df_eval["pair_id"] != row["pair_id"]]

    # TODO: generate real augmented samples
    
    # For now we will use samples from the validation set with the same story as proxy augmented samples
    if( args.augment_on_single_story ):
        # For multi-section target story, check if cor_section of the validation set sample is a subset of cor_section of the target story
        row_cor_section = set(str(row["cor_section"]).split(","))
        df_eval["cor_section_match"] = df_eval["cor_section"].apply(lambda x: set(str(x).split(",")).issubset(row_cor_section) )
        df_samples = df_eval[ ( df_eval["source_title"] == row["source_title"] ) & ( df_eval["cor_section_match"] ) ]
    else:
        df_samples = df_eval[(df_eval["source_title"] == row["source_title"])]
    
    samples = []
    for _, sample_row in df_samples.iterrows():
        samples.append((get_story(sample_row, story_map), sample_row["answer"], sample_row["question"]))
    random.shuffle(samples)

    return samples


def get_incontext_samples_augment(row, story_map, df_eval, args):
    # Choose incontext samples from augmented samples
    samples = get_augmented_samples(row, story_map, df_eval, args)
    
    return samples[:args.num_incontext_ex]


def get_incontext_samples(row, story_map, df_train, df_eval, args, device, embedder=None, corpus_embeddings=None, corpus=None):
    retrieval_query = None
    # Remove current row from df_train based on column pair_id, this should not matter since current row is from df_eval
    df_train = df_train[df_train["pair_id"] != row["pair_id"]]

    if( args.incontext_mode == "random" ):
        samples = get_incontext_samples_random(row, story_map, df_train, args)
    elif( args.incontext_mode == "all_types" ):
        samples =  get_incontext_samples_all_types(row, story_map, df_train, args)
    elif( args.incontext_mode == "retrieval_topk" ):
        samples, retrieval_query =  get_incontext_samples_retrieval_topk(row, story_map, df_train, args, device, embedder, corpus_embeddings)
    elif( args.incontext_mode == "retrieval_all_types" ):
        samples, retrieval_query =  get_incontext_samples_retrieval_all_types(row, story_map, df_train, args, device, embedder, corpus_embeddings, corpus)
    elif( args.incontext_mode == "augment" ):
        samples = get_incontext_samples_augment(row, story_map, df_eval, args)
    else:
        raise ValueError(f"Unknown get incontext_samples mode: {args.incontext_mode}")
    
    return samples, retrieval_query