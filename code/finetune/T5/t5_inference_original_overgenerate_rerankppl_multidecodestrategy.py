'''
python -m code.finetune.inference -N t5_small -M t5-small
'''

import random
import numpy as np
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

import sys
sys.path.insert(1, '/home/zw16/Quest_Question_Generation_Challenge/code/utils')
from create_dataset_split import RAW_DIR, save_csv
from t5_finetune import FinetuneT5

os.environ['WANDB_NOTEBOOK_NAME'] = 'T5InferenceFairyTaleQA'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pdb import set_trace

import numpy as np
import random
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(0)

# %%
# load dataset
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
        sec_nums = row['cor_section'].split(',')
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

# Constrcut transformer input 
def construct_transformer_input(story, answer, choice=1):
    inps = []
    if choice == 1:
        prefix = 'Generate question from story and answer: '
    elif choice == 2:
        prefix = 'Generate question: '
    else:
        prefix = ''
    for stry, ans in zip(story, answer):
        transformer_input = prefix + ' The story is ' + stry + ' The answer is ' + ans 
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

def compute_perplexity(logits, labels):
    """
    Compute the perplexity using logits (dimension = (seq_len, vocab_size) 
    and labels (dimension = (seq_len))
    """
    return torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='mean'))

# Generate from saved model
def get_generation(model, val_dataloader, force_words_ids, decoding_strategy, num_beams=3, prob_p=0.9, temp=1, K=6, alpha=0.6, num_samples=10):
    val_outputs = []
    val_outputs_ppl = []
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        # TODO: Force ? to occur in the sentence
        if decoding_strategy == 'B': # Beam search
            generation = model.generate(val_input_ids, force_words_ids=force_words_ids, 
                                        num_beams = num_beams, temperature=temp,
                                        max_new_tokens=64, 
                                        output_scores=True, return_dict_in_generate=True)
        elif decoding_strategy == 'N': # Nucleus Sampling
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        top_p=prob_p, temperature=temp,
                                        num_return_sequences=num_samples, 
                                        output_scores=True, return_dict_in_generate=True)
        elif decoding_strategy == 'C': # Contrastive Decoding
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        penalty_alpha=alpha, top_k=K,
                                        num_return_sequences=num_samples, 
                                        output_scores=True, return_dict_in_generate=True)
            
        for idx in range(generation['sequences'].shape[0]):
            gen = generation['sequences'][idx]
            valid_gen_idx = torch.where(gen!=0)[0]
            logits = torch.vstack([generation['scores'][i][idx].unsqueeze(0) for i in valid_gen_idx-1])
            ppl = compute_perplexity(logits, gen[gen!=0])
            assert(torch.isnan(ppl) == False)
            val_outputs.append(gen)
            val_outputs_ppl.append(ppl.item())

    return val_outputs, val_outputs_ppl

def get_preds(tokenizer, generated_tokens):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

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

def set_seed(seed_val = 37):
    # setting the seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-TGU', '--track_gpu_usage', action='store_true', help='Track GPU Usage')
    parser.add_argument('-AVG', '--average_decoding', action='store_true', help='Average Decoding')
    parser.add_argument('-AVGN', '--average_decoding_nseeds', type=int, default=5, help='Fold Number of validation set')
    parser.add_argument('-Fold', '--fold_decoding', action='store_true', help='Average Decoding')
    parser.add_argument('-FN', '--fold_number', type=int, default=0, help='Fold Number of validation set')
    parser.add_argument("-F", "--eval_folder", type=str, default="train_val_split_csv", help="Evaluation Folder where output is saved (testset for testing on test set)")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for passing through the Transformer Model")
    parser.add_argument("-MT", "--model_type", type=str, default="T", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
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

    story_file = '../../../data/original/source_texts.csv'
    story_df = pd.read_csv(story_file)
    # Train-Val split
    train_file = '../../../data/train_val_split_csv/train.csv'
    train_df = pd.read_csv(train_file)
    if args.eval_folder == 'train_val_split_csv':
        val_file = '../../../data/train_val_split_csv/val.csv'
        filetype = 'train'
    elif args.eval_folder == 'data_augmentation':
        val_file = '../../../data/train_val_split_csv/train.csv'
        filetype = 'train'
    elif args.eval_folder == 'testset':
        val_file = '../../../data/original/test.csv'
        filetype = 'test'

    if args.fold_decoding:
        val_file = './../../data/score_prediction/qg_model/fold_{:d}/train_val_split_csv/val.csv'.format(args.fold_number)
        filetype = 'train'


    val_df = pd.read_csv(val_file)


    train_story, train_answer, train_question = get_parallel_corpus(train_df, story_df)
    val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df, filetype=filetype)

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
    train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(tokenizer, train_inps, train_question)
    val_input_ids, val_attention_mask, val_labels = get_transformer_encoding(tokenizer, val_inps, val_question)
    print('Tokenized Data!')

    # %%
    train_dataset = FairyDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    print('Created Pytorch Dataset')

    # %%
    batch_size = args.batch_size
    train_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, val_dataset, datatype='val')
    print('Loaded Dataloader!')

    # %%
    # Load the Generative Head 
    # search for ckpt file
    search_dir = os.path.join('../Checkpoints', args.run_name)
    for file in os.listdir(search_dir):
        name, ext = os.path.splitext(file)
        if ext == '.ckpt':
            ckpt_file = os.path.join(search_dir, file)

    print('ckpt_file', ckpt_file)
    # model_pl = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name)
    model = FinetuneT5.load_from_checkpoint(ckpt_file, model_type = args.model_type).model.to(device)
    print('Successfully loaded the saved checkpoint!')

    force_tokens = ['?']
    force_words_ids = tokenizer(force_tokens, add_special_tokens=False).input_ids

    # NOTE: Track GPU Utilization
    if args.track_gpu_usage:
        print('Tracking GPU Usage')
        monitor = Monitor(10)

    # TODO: Implement Average (Multiple) Decoding
    if args.average_decoding:
        if args.average_decoding_nseeds == 5:
            seed_vals = [37, 49, 105, 1, 25]
        elif args.average_decoding_nseeds == 10:
            seed_vals = [37, 49, 105, 1, 25, 2012, 2016, 2020, 2022, 2023]
            assert(len(seed_vals) == 10)
        elif args.average_decoding_nseeds == 25:
            seed_vals = [37, 49, 105, 1, 25, 2012, 2016, 2020, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038]
            assert(len(seed_vals) == 25)
        elif args.average_decoding_nseeds == 50:
            seed_vals = [37, 49, 105, 1, 25, 2012, 2016, 2020, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063]
            assert(len(seed_vals) == 50)
        else:
            raise Exception('Invalid number of seeds for average decoding')
    else:
        seed_vals = [37]


    for ctr in ['B', 'C', 'N']:
        seed = 37
        # Set seed
        set_seed(seed_val = seed)
        print('Begining Generation')
        val_outputs, val_outputs_ppl = get_generation(model, valid_dataloader, force_words_ids, 
                        ctr,
                        args.num_of_beams, 
                        args.p_sampling, args.temperature, 
                        args.top_K, args.alpha,
                        args.num_of_samples)
        print('Done Generating!')

        if args.track_gpu_usage:
            monitor.stop()

        print('Begining Decoding')
        val_preds = get_preds(tokenizer, val_outputs)
        print('Done Decoding!')

        if args.fold_decoding:
            split_data = args.run_name.split('/')
            main_dir, sub_dir = split_data[0], split_data[1]
            print('main_dir, sub_dir:', main_dir, sub_dir)
        else:
            sub_dir = args.run_name

        # NOTE: Saving val_preds
        if args.eval_folder != 'testset':
            val_df['prompt'] = val_inps
            val_df['question'] = val_question

        if ctr != 'B':
            times = [args.num_of_samples for _ in range(len(val_df))]
            new_val_df = val_df.loc[val_df.index.repeat(times)].reset_index(drop=True)
        else:
            new_val_df = val_df
        save_csv_name = 'multiStrategy_{:s}_{:.2f}_{:.2f}_{:d}'.format(
            sub_dir, args.p_sampling, args.temperature, args.num_of_samples)

        # Save predictions
        preds_df = pd.DataFrame()
        preds_df['pair_id'] = new_val_df['pair_id']
        if args.eval_folder != 'testset':
            preds_df['attribute1'] = new_val_df['attribute1']
            preds_df['local_or_sum'] = new_val_df['local_or_sum']
            preds_df['ex_or_im'] = new_val_df['ex_or_im']
            if args.eval_folder == 'data_augmentation':
                preds_df['source_title'] = new_val_df['source_title']
                preds_df['cor_section'] = new_val_df['cor_section']
                preds_df['answer'] = new_val_df['answer']
            preds_df['prompt'] = new_val_df['prompt']
            preds_df['question'] = new_val_df['question']
        preds_df['generated_question'] = val_preds
        preds_df['ppl'] = val_outputs_ppl

        # Create output directory 
        output_path = os.path.join(RAW_DIR, "{}/results".format(args.eval_folder), save_csv_name) 
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if args.fold_decoding:
            output_path = os.path.join(output_path, main_dir)
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        print(output_path, save_csv_name)
        
        # Save the predictions
        save_csv(preds_df, ctr, output_path)