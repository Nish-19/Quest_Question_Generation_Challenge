"""
Batch collator for language model training.
"""

import torch

from code.gpt3.prepare_dataset import clean_str


PROMPT_END_TOKEN = "The question is:"
SEP_TOKEN = "\n"
# COMPLETION_END_TOKEN not required since all questions are already suffixed with a question mark
COMPLETION_END_TOKEN = ""
INSTRUCTIONS = "Write a question based on the story and answer below."


def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")


def get_story_input(title, section, story_map):
    story_txt = story_map[title][section]

    return ("The story is: " + story_txt)


def get_answer_input(answer_txt):
    
    return ("The answer is: " + answer_txt)


def get_question_completion(question_txt):
    
    return ("The question is: " + question_txt)


def process_multiple_sections(cor_section):
    cor_section = int(str(cor_section).split(",")[0])

    return cor_section


def create_prompt(row, story_map, add_instructions=True):
    story = story_map[clean_str(row["source_title"])][process_multiple_sections(row["cor_section"])]
    
    # Suffix prompt with PROMPT_END_TOKEN
    # Separate story and answer with SEP_TOKEN
    if( add_instructions ):
        # For prompting a non finetuned model, i.e., zero shot or few shot learning on off-the-shelf models, add instructions as prefix
        prompt = f"""{INSTRUCTIONS}{SEP_TOKEN}The story is: {story}{SEP_TOKEN}The answer is: {clean_str(row['answer'])}{SEP_TOKEN}{PROMPT_END_TOKEN}"""
    else:
        # If using a finetuned model, no instructions are needed as prefix
        prompt = f"""The story is: {story}{SEP_TOKEN}The answer is: {clean_str(row['answer'])}{SEP_TOKEN}{PROMPT_END_TOKEN}"""

    return prompt


def create_completion(row):
    # Precede completion text with a single space
    # Suffix completetion with END_TOKEN
    completion = f""" {clean_str(row['question'])}{COMPLETION_END_TOKEN}"""

    return completion


def find_subsequence_index(subsequence, sequence):
    
    return [(i, i+len(subsequence)) for i in range(len(sequence)) if sequence[i:i+len(subsequence)] == subsequence]


class CollateWraperParent:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.params = params


class CollateWraperGenerative(CollateWraperParent):
    def __init__(self, tokenizer, params, story_map, lm_loss_location="question"):
        super().__init__(tokenizer, params)
        self.params = params
        self.story_map = story_map
        self.lm_loss_location = lm_loss_location
    

    def __call__(self, batch):
        # Construct text features
        features = []
        for row in batch:
            prompt = create_prompt(row, self.story_map, self.params.add_instructions)
            completion = create_completion(row)
            features.append(prompt + completion)
        # Tokenize
        inputs = tokenize_function(self.tokenizer, features)

        # Labels are only the part we wish to generate, i.e., completion
        inputs["labels"] = inputs["input_ids"].detach().clone()
        # TODO: add mask = -100 to labels for tokens that should not be predicted
        if( self.lm_loss_location == "question" ):
            input_ids_list = inputs["labels"].tolist()
            max_len = len(input_ids_list[0])
            # 25 corresponds to : in vocab https://huggingface.co/gpt2/raw/main/vocab.json
            mask_end_index = [(max_len - 1 - tokens[::-1].index(25)) for tokens in input_ids_list]
            mask_end_index = torch.as_tensor(mask_end_index)
            mask = torch.arange(max_len)[None, :] <= mask_end_index[:, None]
            inputs["labels"] = inputs["labels"].masked_fill(mask, -100)
            # TODO: better way to do masking?
            #max_len = len()
            #mask_end_index = [tokens.index(25) for tokens in inputs["input_ids"].tolist()]
            #print((inputs["input_ids"] == 25).nonzero(as_tuple=True))
            #mask_end_indices = (inputs["input_ids"] == 25).nonzero(as_tuple=True)[1]
            #mask = torch.arange(maxlen)[None, :] < lengths[:, None]
        elif( self.lm_loss_location == "question_answer" ):
            #mask_end_index = [tokens.index(25) for tokens in inputs["input_ids"]]
            pass
        elif( self.lm_loss_location == "all" ):
            pass

        #print(inputs)
        #for ids in inputs["input_ids"]:
        #    print(self.tokenizer.decode(ids))

        return {
            "inputs" : inputs
        }