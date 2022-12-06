'''
python -m code.ans_generation.inference \
-MT T -N t5_small -M t5-small
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
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5Tokenizer
from transformers import BartTokenizer

from code.utils.create_dataset_split import RAW_DIR, save_csv
from code.finetune.finetune import FinetuneTransformer

# setting the seed
seed_val = 37
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


os.environ['WANDB_NOTEBOOK_NAME'] = 'FinetuneTransformerForAnswerGeneration'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# load dataset
def clean_str(text):
    # Replace double quotes with single quotes
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Tokenization
def get_transformer_encoding(tokenizer, prompts, gen_answers):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    max_source_length, max_target_length = 512, 512

    inp_encoding = tokenizer(prompts, padding='longest', 
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt"
                    )
    input_ids, attention_mask = inp_encoding.input_ids, inp_encoding.attention_mask

    target_encoding = tokenizer(gen_answers, padding='longest', 
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
    
def compute_perplexity(model, batch_encodings):
    perplexity = []
    max_length = model.config.n_positions
    stride = max_length
    
    for batch in tqdm(batch_encodings):
        prompt_ip_ids, gen_ans = batch['input_ids'].to(device), batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(prompt_ip_ids, labels=gen_ans)
        # seq_len = encodings['input_ids'].size(1)

        # nlls = []
        # prev_end_loc = 0
        # for begin_loc in tqdm(range(0, seq_len, stride)):
        #     end_loc = min(begin_loc + max_length, seq_len)
        #     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        #     input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
        #     target_ids = input_ids.clone()
        #     target_ids[:, :-trg_len] = -100

            

        #         # loss is calculated using CrossEntropyLoss which averages over input tokens.
        #         # Multiply it with trg_len to get the summation instead of average.
        #         # We will take average over all the tokens to get the true average
        #         # in the last step of this example.
        #         neg_log_likelihood = outputs.loss * trg_len

        #     nlls.append(neg_log_likelihood)

        #     prev_end_loc = end_loc
        #     if end_loc == seq_len:
        #         break

        ppl = torch.exp(outputs.loss)
        perplexity.append(ppl.cpu().clone().numpy().item())

    return perplexity




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
    parser.add_argument("-F", "--eval_folder", type=str, default="data_augmentation", help="Evaluation Folder where output is saved")
    parser.add_argument("-EF", "--gen_filename", type=str, default="answer_nucleus_flan_t5_large_0.95_1.20.csv", help="Evaluation filename")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for passing through the Transformer Model")
    parser.add_argument("-MT", "--model_type", type=str, default="t", help="T for T5 and B for BART")
    parser.add_argument("-N", "--run_name", type=str, default="t5-small", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-MN", "--model_name", default="t5-small", help="Variant of the Transformer model for finetuning")
    params = parser.parse_args()
    
    return params


# %%
if __name__=='__main__':
    args = add_params()

    ques_path = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    gen_file = os.path.join(ques_path, args.gen_filename)
    gen_df = pd.read_csv(gen_file)

    if args.model_type == 'T':
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, use_fast=True)
    elif args.model_type == 'B':
        tokenizer = BartTokenizer.from_pretrained(args.model_name, use_fast=True)
    else:
        print('Wrong model type - either T or B only')

    # %%
    print('Tokenizing Data:')
    prompt_input_ids, prompt_attention_mask, ans_labels = get_transformer_encoding(tokenizer, gen_df['prompt'].tolist(), gen_df['generated_answer'].tolist())
    print('Tokenized Data!')

    # %%
    dataset = FairyDataset(prompt_input_ids, prompt_attention_mask, ans_labels)
    print('Created Pytorch Dataset')

    # %%
    batch_size = args.batch_size
    dataloader = get_dataloader(batch_size, dataset, datatype='val')
    print('Loaded Dataloader!')

    # %%
    # Load the Generative Head 
    # search for ckpt file
    search_dir = os.path.join('./code/ans_generation/Checkpoints', args.run_name)
    for file in os.listdir(search_dir):
        ckpt_file = os.path.join(search_dir, file)
    # model_pl = FinetuneTransformer(model_type = args.model_type, model_name = args.model_name)
    model = FinetuneTransformer.load_from_checkpoint(ckpt_file, model_type = args.model_type).model.to(device)
    print('Successfully loaded the saved checkpoint!')

    # NOTE: Track GPU Utilization
    if args.track_gpu_usage:
        print('Tracking GPU Usage')
        monitor = Monitor(10)
    
    # TODO: Driver code for perplexity calculation
    print('Beginning Perpleixty Calculation')
    perplexity = compute_perplexity(model, dataloader)
    print(perplexity)
    print('Ending Perplexity Calculation')

    if args.track_gpu_usage:
        monitor.stop()

    # Save predictions
    preds_df = pd.DataFrame()
    preds_df['pair_id'] = gen_df['pair_id']
    preds_df['generated_answer'] = gen_df['generated_answer']
    preds_df['perplexity'] = perplexity
    save_csv_name = 'ppl_' + os.path.splitext(args.gen_filename)[0]

    output_path = os.path.join(RAW_DIR, "results/{}".format(args.eval_folder))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_csv(preds_df, save_csv_name, output_path)