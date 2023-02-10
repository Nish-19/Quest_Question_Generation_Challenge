"""
3 batch size x 4 grad acc step = 12 = 0.8854 = sleek-aardvark-135 = epoch=0-step=500.ckpt
2 batch size x 8 grad acc step = 16 = 0.8629 = swept-star-137 = epoch=0-step=375.ckpt
3 batch size x 6 grad acc step = 18 = 0.8428 = misunderstood-deluge-136 = epoch=0-step=333.ckpt


python3 -m code.decoder.generate \
    -MN "google/flan-t5-xl" \
    -N epoch=0-step=375.ckpt \
    --eval_type "local_val" \
    --decoding_strategies "C" \
    --seeds "21" \
    -NS 20 \
    --batch_size_eval 1 \
    --training_strategy "DS" \
    -W "swept-star-137"

python3 -m code.decoder.generate \
    -MN "google/flan-t5-large" \
    -N epoch=0-step=188.ckpt \
    --eval_type "local_val" \
    --decoding_strategies "C-N" \
    --seeds "21" \
    -NS 20 \
    --batch_size_eval 1 \
    --debug

python3 -m code.decoder.generate \
    -MN "google/flan-t5-large" \
    -N epoch=0-step=188.ckpt \
    --eval_type "leaderboard_public_test" \
    --eval_folder "original" \
    --eval_filename "test.csv" \
    --decoding_strategies "C" \
    --seeds "21" \
    -NS 2 \
    --debug
"""
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import pathlib
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

from code.decoder.t5_finetune import FinetuneT5
from code.decoder.rank import rank
from code.decoder.utils import get_parallel_corpus, construct_transformer_input_old_vary, get_transformer_encoding, FairyDataset, get_dataloader
from code.utils.create_dataset_split import load_df, RAW_DIR, save_csv
from code.t5.evaluate import save_params


def add_params():
    parser = argparse.ArgumentParser()

    # Model specification
    parser.add_argument("-W", "--wandb_name", type=str, default="epoch=0-step=2.ckpt", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-N", "--run_name", type=str, default="epoch=0-step=2.ckpt", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-MT", "--model_type", type=str, default="T", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", default="google/flan-t5-large", help="Variant of the transformer model for finetuning")
    parser.add_argument("--model_folder", type=str, default="best_model", help="Finetuned model folder relative to saved models dir")
    # Data params
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="folds/seed_21/train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="val.csv", help="Evaluation filename")
    # Dataloader params
    parser.add_argument('--batch_size_eval', default=8, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    parser.add_argument("-PC", "--prefix_choice", type=int, default=1, help="Choice of prefix used for the input construction - 1, 2, 3")
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


def load_data(args, tokenizer):
    # Load stories
    folder = os.path.join(RAW_DIR, "original")
    story_df = load_df("source_texts.csv", folder)
    # Load evaluation set
    if args.eval_type == 'local_val':
        folder = os.path.join(RAW_DIR, args.eval_folder)
        val_filename = args.eval_filename
        filetype = 'train'
    else:
        folder = os.path.join(RAW_DIR, args.eval_folder)
        val_filename = args.eval_filename
        filetype = 'test'
    nrows = 8 if args.debug else None
    val_df = load_df(val_filename, folder, nrows)
    # Prepare dataloader
    val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df, filetype=filetype)
    val_inps = construct_transformer_input_old_vary(val_story, val_answer, args.prefix_choice)
    val_input_ids, val_attention_mask, val_labels = get_transformer_encoding(tokenizer, val_inps, val_question)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    val_dataloader = get_dataloader(args.batch_size_eval, val_dataset, datatype='val')

    return val_dataloader, val_df


def load_model(args, device, saved_models_dir):
    if( args.training_strategy == "DS" ):
        ckpt_file = os.path.join(saved_models_dir, args.wandb_name, "flan-t5-xl", args.run_name, "huggingface_model")
    else:
        ckpt_file = os.path.join(saved_models_dir, "best_model", args.run_name)
    #model = FinetuneT5.load_from_checkpoint(ckpt_file, model_type = args.model_type).model.to(device)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_file).to(device)
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
    args = add_params()
    # Get saved models dir
    if ( torch.cuda.is_available() ):
        saved_models_dir = "/work/nigel_umass_edu/qg_challenge/finetune/checkpoints_new/" if args.training_strategy == "DS" else "/work/nigel_umass_edu/qg_challenge/saved_models/"
    else:
        saved_models_dir = "../saved_models/"
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'
    model, tokenizer = load_model(args, device, saved_models_dir)
    val_dataloader, val_df = load_data(args, tokenizer)
    val_df = generate_wrapper(model, tokenizer, val_dataloader, val_df, args, device)

    # Get top-10 generated questions according to scores for each pair id
    df_submission, df_test = rank(val_df)
    # Save top-10 generated questions for each pair id in submission format
    save_submission(df_submission, args, df_test)


if __name__ == '__main__':
    main()