'''
python -m code.finetune.inference -N t5_small -M t5-small
'''

from tqdm import tqdm
import GPUtil
from threading import Thread
import time
import argparse
import re
import wandb, os
from collections import defaultdict
import statistics
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer
from transformers import BartTokenizer
from sklearn.metrics import classification_report

from code.utils.create_dataset_split import RAW_DIR, save_csv
from code.finetune.finetune import FinetuneTransformer

os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneTransformer'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# load dataset
def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_parallel_corpus(ip_df, story_df):
    # hash stories and sections
    story_sec_hash = defaultdict(dict)
    for i, row in story_df.iterrows():
        story_sec_hash[row['source_title']][row['cor_section']] = clean_str(row['text'])
    
    story, answer, question, attr = [], [], [], []
    for i, row in ip_df.iterrows():
        sec_nums = row['cor_section'].split(',')
        story_str = ''
        for sec_num in sec_nums:
            story_str += story_sec_hash[row['source_title']][int(sec_num)]
        story.append(story_str)
        answer.append(clean_str(row['answer']))
        question.append(clean_str(row['question']))
        attr.append(clean_str(row['attribute1']))
    
    return story, answer, question, attr

# Constrcut transformer input 
def construct_transformer_input(story, answer, choice=1):
    inps = []
    if choice == 1:
        prefix = 'Generate attriburte and question from story and answer: '
        suffix = ''
    elif choice == 2:
        prefix = 'Generate attribute and question: '
        suffix = ''
    elif choice == 3:
        prefix = ''
        suffix = ''
    elif choice == 4:
        prefix = 'Generate attribute and question from story and answer: '
        suffix = '\nThe question is:'
    for stry, ans in zip(story, answer):
        transformer_input = prefix + '\nThe story is ' + stry + '\nThe answer is ' + ans + suffix
        inps.append(transformer_input)
    return inps

# Tokenization
def get_transformer_encoding(tokenizer, transformer_inputs, question):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 1024, 128

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
    if type == 'train':
        return DataLoader(dataset=dataset, shuffle=True, batch_size = batch_size)
    else:
        return DataLoader(dataset=dataset, batch_size = batch_size)

# Generate from saved model
def get_generation(model, val_dataloader, force_words_ids, decoding_strategy='B', num_beams=3, prob_p=0.9, temp=1, K=6, alpha=0.6, num_samples=10):
    val_outputs = []
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        # TODO: Force ? to occur in the sentence
        if decoding_strategy == 'B': # Beam search
            generation = model.generate(val_input_ids, force_words_ids=force_words_ids, 
                                        num_beams = num_beams, temperature=temp,
                                        max_new_tokens=64)
        elif decoding_strategy == 'N': # Nucleus Sampling
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        top_p=prob_p, temperature=temp,
                                        num_return_sequences=num_samples)
        elif decoding_strategy == 'C': # Contrastive Decoding
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        penalty_alpha=alpha, top_k=K,
                                        num_return_sequences=num_samples)
        else:
            generation = model.generate(val_input_ids, temperature=temp, max_new_tokens=64)
        for gen in generation:
            val_outputs.append(gen)
    return val_outputs

def get_preds(tokenizer, generated_tokens):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

def split_preds(preds):
    all_attr, all_ques = [], []
    for pred in preds:
        pre_attr, pre_ques = pred.split('The question is:')
        pre_attr = pre_attr.split('The attribute is:')[1]
        pre_attr, pre_ques = clean_str(pre_attr), clean_str(pre_ques)
        all_attr.append(pre_attr)
        all_ques.append(pre_ques)
    return all_attr, all_ques

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-TGU', '--track_gpu_usage', action=argparse.BooleanOptionalAction, help='Track GPU Usage')
    parser.add_argument("-F", "--eval_folder", type=str, default="train_val_split_csv", help="Evaluation Folder where output is saved")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for passing through the Transformer Model")
    parser.add_argument("-MT", "--model_type", type=str, default="t", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
    parser.add_argument('-DS', '--decoding_strategy', type=str, default="G", help='Specify the decoding strategy (B-Beam Search, N-Nucleus sampling, G-Greedy, C-Contrastive)')
    parser.add_argument("-PS", "--p_sampling", type=float, default=0.9, help="Value of P used in the P-sampling")
    parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature for softmax decoding")
    parser.add_argument("-K", "--top_K", type=int, default=4, help="Value of K used for contrastive decoding")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.6, help="Value of alpha used for contrastive decoding")
    parser.add_argument("-NS", "--num_of_samples", type=int, default=10, help="Number of samples to generate when using sampling")
    parser.add_argument('-NB', '--num_of_beams', type=int, default=3, help="Number of beams for decoding")
    parser.add_argument("-PC", "--prefix_choice", type=int, default=1, help="Choice of prefix used for the input construction - 1, 2, 3")
    params = parser.parse_args()
    
    return params


# %%
if __name__=='__main__':
    args = add_params()

    story_file = './data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = './data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    if args.eval_folder == 'train_val_split_csv':
        val_file = './data/train_val_split_csv/val.csv'
    elif args.eval_folder == 'data_augmentation':
        val_file = './data/train_val_split_csv/train.csv'
    val_df = pd.read_csv(val_file)

    train_story, train_answer, train_question, train_attr = get_parallel_corpus(train_df, story_df)
    val_story, val_answer, val_question, val_attr = get_parallel_corpus(val_df, story_df)

    # %%
    train_inps = construct_transformer_input(train_story, train_answer, args.prefix_choice)
    val_inps = construct_transformer_input(val_story, val_answer, args.prefix_choice)

    if args.model_type == 'T':
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    elif args.model_type == 'B':
        tokenizer = BartTokenizer.from_pretrained(args.model_name)
    else:
        print('Wrong model type - either T or B only')

    # %%
    # train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(tokenizer, train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_transformer_encoding(tokenizer, val_inps, val_question)
    print('Tokenized Data!')

    # %%
    # train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')

    # %%
    batch_size = args.batch_size
    # train_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    # %%
    # Load the Generative Head 
    # search for ckpt file
    search_dir = os.path.join('./code/finetune_mtask/Checkpoints', args.run_name)
    for file in os.listdir(search_dir):
        ckpt_file = os.path.join(search_dir, file)
    print('ckpt_file', ckpt_file)
    # model_pl = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name)
    model = FinetuneTransformer.load_from_checkpoint(ckpt_file, model_type = args.model_type).model.to(device)
    print('Successfully loaded the saved checkpoint!')

    force_tokens = ['?']
    force_words_ids = tokenizer(force_tokens, add_special_tokens=False).input_ids

    # NOTE: Track GPU Utilization
    if args.track_gpu_usage:
        print('Tracking GPU Usage')
        monitor = Monitor(10)

    print('Begining Generation')
    val_outputs = get_generation(model, valid_dataloader, force_words_ids, 
                    args.decoding_strategy, args.num_of_beams, 
                    args.p_sampling, args.temperature, 
                    args.top_K, args.alpha,
                    args.num_of_samples)
    print('Done Generating!')

    if args.track_gpu_usage:
        monitor.stop()

    print('Begining Decoding')
    val_preds = get_preds(tokenizer, val_outputs)
    print('Done Decoding!')

    # NOTE: Split attribute and question
    attr_preds, ques_preds = split_preds(val_preds)


    # NOTE: Saving val_preds
    val_df['prompt'] = val_inps
    val_df['question'] = val_question
    if args.decoding_strategy == 'N':
        times = [args.num_of_samples for _ in range(len(val_df))]
        new_val_df = val_df.loc[val_df.index.repeat(times)].reset_index(drop=True)
        save_csv_name = 'nucleus_{:s}_{:.2f}_{:.2f}_{:d}'.format(args.run_name, args.p_sampling, args.temperature, args.num_of_samples)
    elif args.decoding_strategy == 'C':
        times = [args.num_of_samples for _ in range(len(val_df))]
        new_val_df = val_df.loc[val_df.index.repeat(times)].reset_index(drop=True)
        save_csv_name = 'contrastive_{:s}_{:d}_{:.2f}_{:d}'.format(args.run_name, args.top_K, args.alpha, args.num_of_samples)
    else:
        new_val_df = val_df
        save_csv_name = args.run_name
    
    # NOTE: Report attribute pred statistics
    print(classification_report(new_val_df['attribute1'].apply(clean_str), attr_preds))

    # Save predictions
    preds_df = pd.DataFrame()
    preds_df['pair_id'] = new_val_df['pair_id']
    preds_df['attribute1'] = new_val_df['attribute1']
    preds_df['local_or_sum'] = new_val_df['local_or_sum']
    preds_df['ex_or_im'] = new_val_df['ex_or_im']
    if args.eval_folder == 'data_augmentation':
        preds_df['source_title'] = new_val_df['source_title']
        preds_df['cor_section'] = new_val_df['cor_section']
        preds_df['answer'] = new_val_df['answer']
    preds_df['prompt'] = new_val_df['prompt']
    preds_df['question'] = new_val_df['question']
    preds_df['predicted attribute'] = attr_preds
    preds_df['generated_question'] = ques_preds

    output_path = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_csv(preds_df, save_csv_name, output_path)