import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import pathlib
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration


from train import FinetuneTransformer
from rank import rank
from train import get_transformer_encoding, FairyDataset, get_dataloader


def save_csv(df, filename, dirname):
    filepath = os.path.join(dirname, filename + ".csv")
    df.to_csv(filepath, encoding='utf-8', index=False)

def add_params():
    parser = argparse.ArgumentParser()

    # Model specification
    parser.add_argument("-N", "--run_name", type=str, default="reft_flan_t5_large_nodup_selemaugment_full_ga", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-MN", "--model_name", default="google/flan-t5-large", help="Variant of the transformer model for finetuning")
    parser.add_argument("--model_folder", type=str, default="best_model", help="Finetuned model folder relative to saved models dir")
    # Data params
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="folds/seed_21/train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    # Dataloader params
    parser.add_argument('--batch_size_eval', default=8, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    # Decoding params
    parser.add_argument("-NS", "--num_of_samples", type=int, default=10, help="Number of samples to generate when using sampling")
    parser.add_argument("-PS", "--p_sampling", type=float, default=0.9, help="Value of P used in the P-sampling")
    parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature for softmax decoding")
    parser.add_argument("-K", "--top_K", type=int, default=4, help="Value of K used for contrastive decoding")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.6, help="Value of alpha used for contrastive decoding")
    parser.add_argument('-NB', '--num_of_beams', type=int, default=10, help="Number of beams for decoding")
    parser.add_argument('--decoding_strategies', default="C", type=str, help='Random seed') 
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seeds', default="21", type=str, help='Random seed') 
    parser.add_argument('-TS', '--training_strategy', type=str, default="DP", help="DP for dataparalle and DS for deepspeed")
    params = parser.parse_args()
    
    return params

def set_random_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True

def load_data(test_inps, batch_size, tokenizer):
    test_input_ids, test_attention_mask, _ = get_transformer_encoding(tokenizer, test_inps)
    test_dataset = FairyDataset(test_input_ids, test_attention_mask)
    test_dataloader = get_dataloader(batch_size, test_dataset, datatype='val')
    return test_dataloader 

def load_model(args, device, saved_models_dir):
    if( args.training_strategy == "DS" ):
        ckpt_file = os.path.join(saved_models_dir, args.wandb_name, "flan-t5-xl", args.run_name, "huggingface_model")
    else:
        # NOTE: Nischal's code for loading the model
        search_dir = os.path.join(saved_models_dir, args.run_name)
        for file in os.listdir(search_dir):
            name, ext = os.path.splitext(file)
            if ext == '.ckpt':
                ckpt_file = os.path.join(search_dir, file)

    model = FinetuneTransformer.load_from_checkpoint(ckpt_file).model.to(device)
    
    # model = T5ForConditionalGeneration.from_pretrained(ckpt_file).to(device) # NOTE: Does not work
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model.eval()

    return model, tokenizer

def generate(device, model, val_dataloader, force_words_ids, decoding_strategy, num_beams=10, prob_p=0.9, temp=1, K=4, alpha=0.6, num_samples=10):
    val_outputs = []
    val_outputs_ppl = []
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        # TODO: Force ? to occur in the sentence
        if decoding_strategy == 'B': # Beam search
            generation = model.generate(val_input_ids, force_words_ids=force_words_ids, 
                                        num_beams = num_beams, temperature=temp,
                                        num_return_sequences=num_samples, 
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

def compute_perplexity(logits, labels):
    """
    Compute the perplexity using logits (dimension = (seq_len, vocab_size) 
    and labels (dimension = (seq_len))
    """
    return torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='mean'))


def get_preds(tokenizer, generated_tokens):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

def generate_wrapper(model, tokenizer, val_dataloader, val_df, args, device):
    force_tokens = ['?']
    force_words_ids = tokenizer(force_tokens, add_special_tokens=False).input_ids
    
    all_df = []
    for decoding_strategy in args.decoding_strategies.split('-'):
        for seed in args.seeds.split('-'):
            val_df_curr = val_df.copy()
            set_random_seed(int(seed))
            val_outputs, val_outputs_ppl = generate(device, model, val_dataloader, force_words_ids, 
                                                    decoding_strategy,
                                                    args.num_of_beams, 
                                                    args.p_sampling, args.temperature, 
                                                    args.top_K, args.alpha,
                                                    args.num_of_samples)
            val_preds = get_preds(tokenizer, val_outputs)

            val_preds = [val_preds[x:x+args.num_of_samples] for x in range(0, len(val_preds), args.num_of_samples)]
            val_outputs_ppl = [val_outputs_ppl[x:x+args.num_of_samples] for x in range(0, len(val_outputs_ppl), args.num_of_samples)]
            #print(val_preds)
            #print(val_outputs_ppl)
            val_df_curr['generated_question'] = val_preds
            val_df_curr['score'] = val_outputs_ppl
            all_df.append(val_df_curr)

    all_df = pd.concat(all_df)
    return all_df

def save_submission(df_submission, args, df_debug=None):
    #identifier = f"run-{args.run_name.split('.')[0]}_decoding-{args.decoding_strategies}_seed-{args.seeds}_nsamples-{args.num_of_samples}_time-{time.strftime('%Y%m%d-%H%M%S')}"
    identifier = f"run-{args.wandb_name}_decoding-{args.decoding_strategies}_seed-{args.seeds}_nsamples-{args.num_of_samples}_time-{time.strftime('%Y%m%d-%H%M%S')}"
    # Save submission to csv
    folder = os.path.join(RAW_DIR, "results", args.eval_folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    if( args.eval_type == "local_val"):
        filename = identifier + "_local_debug"
        save_csv(df_debug, filename, folder)
        filename = identifier + "_local"
        save_csv(df_submission, filename, folder)
    else:
        df_submission = df_submission.rename(columns={"generated_question_original": "generated_question"})
        filename = identifier + "_leaderboard_debug"
        save_csv(df_submission, filename, folder)
        # Save in leaderboard format
        filename = identifier + "_leaderboard"
        save_csv(df_submission[["pair_id", "generated_question"]], filename, folder)

    # Save generation parameters in a json file for reference
    save_params(args, filename, folder)

def main():

    # Open settings file
    with open('SETTINGS.json', 'r') as infile:
        json_file = json.load(infile)

    args = add_params()
    # NOTE: Need to have a look
    # # Get saved models dir
    # if ( torch.cuda.is_available() ):
    #     saved_models_dir = "/work/nigel_umass_edu/qg_challenge/finetune/checkpoints_new/" if args.training_strategy == "DS" else "/work/nigel_umass_edu/qg_challenge/saved_models/"
    # else:
    #     saved_models_dir = "../saved_models/"

    # NOTE: Load models
    saved_models_dir = json_file['MODEL_DIR']
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'
    model, tokenizer = load_model(args, device, saved_models_dir)
    print('Loaded tokenizer and model!')

    # NOTE: Load data
    test_file = json_file['TEST_DATA_CLEAN_PATH']
    test_df = pd.read_csv(test_file)

    test_dataloader = load_data(test_df['Transformer Input'].tolist(), args.batch_size_eval, tokenizer)
    print('Loaded the test dataloader')

    # val_df = generate_wrapper(model, tokenizer, val_dataloader, val_df, args, device)

    # # Get top-10 generated questions according to scores for each pair id
    # df_submission, df_test = rank(val_df)
    # # Save top-10 generated questions for each pair id in submission format
    # save_submission(df_submission, args, df_test)


if __name__ == '__main__':
    main()

