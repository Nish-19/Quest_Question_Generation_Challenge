import torch
import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from code.score_prediction.models.flan_t5 import ScorePredictionModelFlanT5Wrapper
from code.score_prediction.models.bert import ScorePredictionModelBertWrapper
from code.gpt3.prepare_dataset import load_stories
from code.t5.train import set_random_seed
from code.utils.utils import agg_all_metrics


RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument("--model_folder", type=str, default="qg_challenge/QGCHAL-36/cross_val_fold_21/best_val_score/", help="GPT-2 model folder relative to saved models dir")
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="score_prediction/score_model/train_val_test_split_json", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="test.csv", help="Evaluation filename")
    parser.add_argument('--max_source_length', default=512, type=int, help='Maximum length of input sequence')
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seed', default=21, type=int, help='Random seed') 

    params = parser.parse_args()
    
    return params


def load_model(args, saved_models_dir, device):
    # Load model wrapper
    if( "flan" in args.lm ):
        model = ScorePredictionModelFlanT5Wrapper(args, device)
    elif( "bert" in args.lm ):
        model = ScorePredictionModelBertWrapper(args, device)
    else:
        raise "Base LM not supported"
    # Load finetuned model weights
    model_folder = os.path.join(saved_models_dir, args.model_folder)
    model.model.tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model.model.model = AutoModelForSequenceClassification.from_pretrained(model_folder, return_dict=True).to(device)
    # Set model to eval mode
    model.set_eval_mode()

    return model


def load_test_set(model, args):
    dir_name = os.path.join(RAW_DIR, args.eval_folder)
    filepath = os.path.join(dir_name, args.eval_filename)
    with open(filepath, "r") as f:
        test_set = json.load(f)
    # Debug with less data
    if(args.debug):
        test_set = test_set[:8]
    model.test_set = test_set


def load_test_data_loader(model, args):
    # TODO p2: story map at test time will be different from train time
    model.test_loader = torch.utils.data.DataLoader(model.testset, collate_fn=model.batch_collator(model.model.tokenizer, model.params, model.story_map), 
                                    batch_size=model.params.batch_size_eval, num_workers=model.params.workers, shuffle=False, drop_last=False)


def predict_bleurt(model, args, device):
    # Evaluation on test set
    test_logs = []

    # Test epoch
    with tqdm(model.test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model.test_step(batch)
            test_logs.append(logs)

    # Aggregate logs across batches
    test_logs = agg_all_metrics(test_logs)

    return test_logs


def pool(predictions, test_set):
    pass

def main():
    # TODO p2: check that picking the best question gives us the max possible bleurt score
    args = add_params()
    # Set saved models dir
    if ( torch.cuda.is_available() ):
        saved_models_dir = "/work/nigel_umass_edu/qg_challenge/saved_models/"
    else:
        saved_models_dir = "../saved_models/"
    # Set random seed if specified
    if args.seed != -1:
        set_random_seed(args.seed)
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'

    # Load model and tokenizer
    model = load_model(args, saved_models_dir, device)
    # Load test set and data loader
    load_test_set(model, args)
    load_test_data_loader(model, args)

    # Run score prediction model for evaluation
    predictions = predict_bleurt(model, args, device)

    # Get top-10 questions
    df_submission = pool(predictions, model.testset)