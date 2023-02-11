import torch
import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
import pathlib
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

from code.fairy_dataset import FairyDataset
from code.rank import rank
#from code.decoder.utils import get_parallel_corpus, construct_transformer_input_old_vary, get_transformer_encoding, FairyDataset, get_dataloader
from code.utils import load_df, save_csv, set_random_seed, get_transformer_encoding, get_dataloader, get_parallel_corpus, construct_transformer_input


def add_params():
    parser = argparse.ArgumentParser()

    # Model specification
    parser.add_argument("-N", "--run_name", type=str, default="devout-jazz", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-MN", "--model_name", default="google/flan-t5-xl", help="Variant of the transformer model for finetuning")
    parser.add_argument("-LEN", "--max_source_length", type=int, default=1024, help="Max input sequence length to for tokenizer")
    # Dataloader params
    parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size')
    # Decoding params
    parser.add_argument("-NS", "--num_of_samples", type=int, default=5, help="Number of samples to generate when using sampling")
    parser.add_argument("-PS", "--p_sampling", type=float, default=0.9, help="Value of P used in the P-sampling")
    parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature for softmax decoding")
    parser.add_argument("-K", "--top_K", type=int, default=4, help="Value of K used for contrastive decoding")
    parser.add_argument("--alpha", type=float, default=0.6, help="Value of alpha used for contrastive decoding")
    parser.add_argument('-NB', '--num_of_beams', type=int, default=10, help="Number of beams for decoding")
    parser.add_argument('--decoding_strategies', default="C", type=str, help='Random seed') 
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seeds', default="21-0-1-2", type=str, help='Random seed') 
    parser.add_argument("--eval_type", type=str, default="leaderboard_public_test", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")

    params = parser.parse_args()
    
    return params


def load_data(args, tokenizer, settings):
    # Load data
    data_dir = settings['RAW_DATA_DIR']
    nrows = 8 if args.debug else None
    val_df = load_df('test.csv', data_dir, nrows=nrows)
    story_df = load_df('source_texts.csv', data_dir)
    # Match story, question and answer
    val_story, val_answer, val_question = get_parallel_corpus(val_df, story_df, filetype="test")
    # Prepare prompts for language model
    val_inps = construct_transformer_input(val_story, val_answer)
    # Tokenize and encode
    val_input_ids, val_attention_mask, val_labels = get_transformer_encoding(tokenizer, val_inps, val_question, max_source_length=args.max_source_length, max_target_length=64)
    val_dataset = FairyDataset(val_input_ids, val_attention_mask, val_labels)
    val_dataloader = get_dataloader(args.batch_size_eval, val_dataset, datatype='val')

    return val_dataloader, val_df


def load_model(args, device, saved_models_dir):
    checkpoint_folder = os.path.join(saved_models_dir, args.run_name, "last.ckpt", "huggingface_model")
    #checkpoint_folder = glob.glob(checkpoint_folder)[0]
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_folder).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model.eval()

    return model, tokenizer


def generate(device, model, val_dataloader, force_words_ids, decoding_strategy, num_beams=10, prob_p=0.9, temp=1, K=4, alpha=0.6, num_samples=10):
    val_outputs = []
    val_outputs_ppl = []
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
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
    # Compute the perplexity using logits (dimension = (seq_len, vocab_size) and labels (dimension = (seq_len))

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
            val_df_curr['generated_question'] = val_preds
            val_df_curr['score'] = val_outputs_ppl
            all_df.append(val_df_curr)

    all_df = pd.concat(all_df)
    
    return all_df


def save_submission(df_submission, args, settings):
    #identifier = f"run-{args.run_name.split('.')[0]}_decoding-{args.decoding_strategies}_seed-{args.seeds}_nsamples-{args.num_of_samples}_time-{time.strftime('%Y%m%d-%H%M%S')}"
    identifier = f"run-{args.run_name}_decoding-{args.decoding_strategies}_seed-{args.seeds}_nsamples-{args.num_of_samples}_time-{time.strftime('%Y%m%d-%H%M%S')}"
    # Save submission to csv
    folder = settings["SUBMISSION_DIR"]
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    if( args.eval_type == "local_val"):
        filename = identifier + "_local"
        save_csv(df_submission, filename, folder)
    else:
        # Save in leaderboard format
        filename = identifier + "_leaderboard"
        save_csv(df_submission[["pair_id", "generated_question"]], filename, folder)


def main():
    args = add_params()
    
    # Load settings 
    args = add_params()
    with open('./SETTINGS.json', 'r') as infile:
        settings = json.load(infile)
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'
    # Load model
    model, tokenizer = load_model(args, device, settings['MODEL_DIR'])
    # Load data
    val_dataloader, val_df = load_data(args, tokenizer, settings)
    val_df = generate_wrapper(model, tokenizer, val_dataloader, val_df, args, device)

    # Get top-10 generated questions according to scores for each pair id
    df_submission = rank(val_df)
    # Save top-10 generated questions for each pair id in submission format
    save_submission(df_submission, args, settings)


if __name__ == '__main__':
    main()