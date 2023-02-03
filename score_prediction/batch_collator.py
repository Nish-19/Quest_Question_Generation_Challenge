import torch

from code.gpt3.prepare_dataset import clean_str
from code.t5.batch_collator import tokenize_function, get_story_txt, CollateWraperParent, SEP_TOKEN


def create_prompt_flan_t5(row, story_map):    
    # Use generated question instead of original question
    question_txt = clean_str(row['generated_question_normalized'])
    answer_txt = clean_str(row['answer'])
    story_txt = get_story_txt(row["source_title"], row["cor_section"], story_map)
    prompt = f"""question: {question_txt}{SEP_TOKEN}answer: {answer_txt}{SEP_TOKEN}context: {story_txt}"""

    return prompt


class CollateWraperScorePredictionBert(CollateWraperParent):
    def __init__(self, tokenizer, params, story_map):
        super().__init__(tokenizer, params)
        self.story_map = story_map


    def __call__(self, batch):
        # Construct text features
        features_1 = []
        features_2 = []
        for row in batch:
            # Use generated question instead of original question
            question_txt = clean_str(row['generated_question_normalized'])
            answer_txt = clean_str(row['answer'])
            story_txt = get_story_txt(row["source_title"], row["cor_section"], self.story_map)
            features_1.append(f"question: {question_txt}")
            features_2.append(f"answer: {answer_txt} [SEP] context: {story_txt}")
         # Tokenize
        inputs = tokenize_function(self.tokenizer, features_1, sentences_2=features_2)
        # Labels are bleurt scores
        labels  =  torch.tensor([row['bleurt_score'] for row in batch])
        """
        # Print sample batch
        print(inputs)
        for ids in inputs["input_ids"]:
            print(self.tokenizer.decode(ids))
        print(labels)
        """

        return {
            "inputs" : inputs,
            "labels" : labels
            }


class CollateWraperScorePredictionFlanT5(CollateWraperParent):
    def __init__(self, tokenizer, params, story_map):
        super().__init__(tokenizer, params)
        self.story_map = story_map


    def __call__(self, batch):
        # Construct text features
        features_input = []
        for row in batch:
            prompt = create_prompt_flan_t5(row, self.story_map)
            features_input.append(prompt)
        # Tokenize
        input_encoding = tokenize_function(self.tokenizer, features_input, max_length=self.params.max_source_length)
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        # Labels are bleurt scores
        labels  =  torch.tensor([row['bleurt_score'] for row in batch])
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