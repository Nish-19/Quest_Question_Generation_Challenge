from code.gpt3.prepare_dataset import clean_str


PROMPT_END_TOKEN = "question:"
COMPLETION_END_TOKEN = ""
INSTRUCTIONS = "write a question from context and answer:"
SEP_TOKEN = " "


def tokenize_function(tokenizer, sentences_1, sentences_2=None, max_length=512):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding="longest", max_length=max_length, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, max_length=max_length, truncation=True, return_tensors="pt")


def get_story_txt(title, sections, story_map):
    sections = str(sections).strip("[]")
    sections = sections.split(",")
    story_txt = " ".join([story_map[title][int(section)] for section in sections])

    return story_txt


def create_prompt(row, story_map, add_instructions=False):    
    # Suffix prompt with PROMPT_END_TOKEN, separate story and answer with SEP_TOKEN
    answer_txt = clean_str(row['answer'])
    story_txt = get_story_txt(row["source_title"], row["cor_section"], story_map)
    if( add_instructions ):
        prompt = f"""{INSTRUCTIONS}{SEP_TOKEN}answer: {answer_txt}{SEP_TOKEN}context: {story_txt}{SEP_TOKEN}{PROMPT_END_TOKEN}"""
    else:
        prompt = f"""answer: {answer_txt}{SEP_TOKEN}context: {story_txt}{SEP_TOKEN}{PROMPT_END_TOKEN}"""

    return prompt


def create_completion(row):
    question_txt = clean_str(row['question'])
    # Suffix completetion with COMPLETION_END_TOKEN which is empty here since all questions end with ?
    completion = f"""{question_txt}{COMPLETION_END_TOKEN}"""

    return completion


class CollateWraperParent:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.params = params


class CollateWraperGenerative(CollateWraperParent):
    def __init__(self, tokenizer, params, story_map):
        super().__init__(tokenizer, params)
        self.story_map = story_map


    def __call__(self, batch):
        # Construct text features
        features_input = []
        features_labels = []
        for row in batch:
            prompt = create_prompt(row, self.story_map, self.params.add_instructions)
            completion = create_completion(row)
            features_input.append(prompt)
            features_labels.append(completion)
        
        # Tokenize
        input_encoding = tokenize_function(self.tokenizer, features_input, max_length=self.params.max_source_length)
        labels_encoding = tokenize_function(self.tokenizer, features_labels, max_length=self.params.max_source_length)

        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        labels = labels_encoding.input_ids
        #for ids in labels:
        #    print(self.tokenizer.decode(ids))

        # Replace padding token id of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        """
        # Print sample batch
        print("Sample batch:")
        for ids in input_ids:
            print(self.tokenizer.decode(ids))
        print("Input ids:", input_ids)
        print("Attention mask:", attention_mask)
        print("Labels:", labels)
        """

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
            }