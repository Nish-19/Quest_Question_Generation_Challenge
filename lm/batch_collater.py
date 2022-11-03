"""
Batch collator for language model training.
"""


def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")


def get_story_input(title, section, story_map):
    story_txt = story_map[title][section]

    return ("Context: " + story_txt)


def get_answer_input(answer_txt):
    return ("Answer: " + answer_txt)


def get_question_prompt(question_txt):
    return ("Question: ")


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
        features = []
        for d in batch:
            story_input = get_story_input(d["source_title"], d["cor_section"], self.story_map)
            answer_input = get_answer_input(d["question_text"]) 
            
            features.append(story_input + " " + answer_input)
        inputs = tokenize_function(self.tokenizer, features)

        # Construct labels
        labels  =  torch.tensor([d['irt_difficulty_norm'] for d in batch])
        inputs['labels'] = labels

        #print(inputs)
        #for ids in inputs["input_ids"]:
        #    print(self.tokenizer.decode(ids))
        #print(labels)

        return {"inputs" : inputs}