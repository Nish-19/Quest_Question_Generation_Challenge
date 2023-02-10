import re
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_parallel_corpus(ip_df, story_df, filetype='train'):
    # hash stories and sections
    story_sec_hash = defaultdict(dict)
    for i, row in story_df.iterrows():
        story_sec_hash[row['source_title']][row['cor_section']] = clean_str(row['text'])
    
    story, answer, question = [], [], []
    for i, row in ip_df.iterrows():
        sec_nums = str(row['cor_section']).split(',')
        story_str = ''
        for sec_num in sec_nums:
            story_str += story_sec_hash[row['source_title']][int(sec_num)]
        story.append(story_str)
        answer.append(clean_str(row['answer']))
        if filetype == 'train':
            question.append(clean_str(row['question']))
        else:
            question.append('None')
    
    return story, answer, question


def construct_transformer_input_old_vary(story, answer, choice=1):
    inps = []
    if choice == 1:
        prefix = 'Generate question from context and answer: '
        suffix = ''
    elif choice == 2:
        prefix = 'Generate question: '
        suffix = ''
    elif choice == 3:
        prefix = ''
        suffix = ''
    elif choice == 4:
        prefix = 'Generate question from context and answer: '
        suffix = '\nThe question is:'
    for stry, ans in zip(story, answer):
        transformer_input = prefix + '\nContext: ' + stry + '\nAnswer: ' + ans + suffix
        inps.append(transformer_input)
    return inps


def get_transformer_encoding(tokenizer, transformer_inputs, question):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 1024, 64

    inp_encoding = tokenizer(transformer_inputs, padding='longest', # longest 
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt"
                    )
    input_ids, attention_mask = inp_encoding.input_ids, inp_encoding.attention_mask

    target_encoding = tokenizer(question, padding='longest', 
                        max_length=max_target_length,
                        truncation=True,
                        return_tensors="pt"
                    )
    labels = target_encoding.input_ids
    # 0 loss for pad tokens
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels


class FairyDataset(Dataset):
    def __init__(self, input_ids, attn_masks, labels):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels
        
    def __getitem__(self, index):
        x = self.input_ids[index]
        y = self.attn_masks[index]
        z = self.labels[index]
        
        return {'input_ids': x, 'attention_mask': y, 'labels':z}
    
    def __len__(self):
        return len(self.input_ids)


def get_dataloader(batch_size, dataset, datatype='train'):
    if datatype == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)