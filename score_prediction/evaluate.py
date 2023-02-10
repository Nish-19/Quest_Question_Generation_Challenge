import torch
import os
import json
import pathlib
import argparse
from tqdm import tqdm
import time
import string
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from code.score_prediction.prepare_dataset_score_model import remove_duplicates
from code.score_prediction.models.flan_t5 import ScorePredictionModelFlanT5Wrapper
from code.score_prediction.models.bert import ScorePredictionModelBertWrapper
from code.gpt3.prepare_dataset import load_stories
from code.t5.train import set_random_seed
from code.utils.utils import agg_all_metrics
from code.utils.create_dataset_split import load_df, save_csv
from code.t5.evaluate import save_params


RAW_DIR = "./data"


def add_params():
    parser = argparse.ArgumentParser()

    # Model specification
    parser.add_argument('--lm', default='google/flan-t5-small', help='Base language model')
    parser.add_argument('--max_source_length', default=512, type=int, help='Maximum length of input sequence')
    parser.add_argument("--model_folder", type=str, default="score_prediction/QGCHAL-109/best_val_score/", help="Finetuned model folder relative to saved models dir")
    # Data params
    parser.add_argument("--eval_type", type=str, default="local_val", choices=["local_val", "leaderboard_public_test"], help="Evaluate local validation set or leaderboard public test set")
    parser.add_argument("--eval_folder", type=str, default="score_prediction/score_model/train_val_test_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument("--eval_filename", type=str, default="test.csv", help="Evaluation filename")
    # Dataloader params
    parser.add_argument('--batch_size_eval', default=64, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers')
    # Misc
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('--seed', default=21, type=int, help='Random seed') 
    # Automatic mixed precision training -> faster training without affecting accuracy on Volta (V100) or Turing (RTX8000) GPUs
    parser.add_argument('--amp', action='store_true', help='apply automatic mixed precision training')

    params = parser.parse_args()
    
    return params


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def load_test_data(model_wrapper, args):
    # TODO P1: check right story map is loaded, at test time story map will be different from train time
    model_wrapper.story_map = load_stories()
    
    # Load test set
    dir_name = os.path.join(RAW_DIR, args.eval_folder)
    df_test = load_df(args.eval_filename, dir_name)
    
    if( args.eval_type == "leaderboard_public_test" ):
        # TODO P1: cross check with Nischal that test set is already normalized and duplicates are removed
        # Normalize generated questions in test set
        df_test['generated_question_normalized'] = df_test['generated_question'].apply(normalize)
        df_test = df_test.rename(columns={"generated_question": "generated_question_original"})
        # Drop duplicate questions using pair id and generated question
        df_test = remove_duplicates(df_test, "test")

        # Add story, section, answer cols
        dir_name = os.path.join(RAW_DIR, "original")
        df_test_original = load_df("test.csv", dir_name)
        df_test = df_test.merge(df_test_original[["pair_id", "source_title", "cor_section", "answer"]], on="pair_id", how="left")

        # Add dummy bleurt score columns
        df_test["bleurt_score"] = 0.0
    
    # Convert to json
    json_out = df_test.to_json(orient="records")
    testset = json.loads(json_out)
    # Debug with less data
    if(args.debug):
        testset = testset[:8]
        df_test = df_test[:8]
    model_wrapper.testset = testset

    return df_test
    

def prepare_test_dataloaders(model_wrapper):
    model_wrapper.test_loader = torch.utils.data.DataLoader(model_wrapper.testset, collate_fn=model_wrapper.batch_collator(model_wrapper.tokenizer, model_wrapper.params, model_wrapper.story_map), 
                                    batch_size=model_wrapper.params.batch_size_eval, num_workers=model_wrapper.params.workers, shuffle=False, drop_last=False)


def predict_bleurt(args, device):
    # Load model model_wrapper
    if( "flan" in args.lm ):
        model_wrapper = ScorePredictionModelFlanT5Wrapper(args, device)
    elif( "bert" in args.lm ):
        model_wrapper = ScorePredictionModelBertWrapper(args, device)
    else:
        raise "Base LM not supported"
    # Load test data
    df_test = load_test_data(model_wrapper, args)
    prepare_test_dataloaders(model_wrapper)

    # Evaluation on test set
    test_logs = []
    # Set model to eval mode
    model_wrapper.set_eval_mode()
    with tqdm(model_wrapper.test_loader, unit="batch", leave=False) as tbatch:
        for batch_num, batch in enumerate(tbatch):
            tbatch.set_description("Batch {}".format(batch_num))
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model_wrapper.test_step(batch)
            test_logs.append(logs)
    # Aggregate logs across batches
    test_logs = agg_all_metrics(test_logs)
    print(test_logs)
    
    return test_logs["score_prediction"]["logits"], df_test


def pool(predictions, df_test):
    assert len(predictions) == len(df_test)
    # Add predictions to df_test
    df_test["bleurt_score_prediction"] = predictions

    # Keep top 10 questions generated according to bleurt score predictions for each pair id
    df_test = df_test.groupby("pair_id").apply(lambda x: x.sort_values("bleurt_score_prediction", ascending=False).head(10))

    return df_test


def save_submission(df_submission, args, run_id):
    # Save submission to csv
    folder = os.path.join(RAW_DIR, "results", args.eval_folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if( args.eval_type == "local_val"):
        filename = run_id + "_" + timestr + "_local"
        save_csv(df_submission, filename, folder)
    else:
        df_submission = df_submission.rename(columns={"generated_question_original": "generated_question"})
        filename = run_id + "_" + timestr + "_leaderboard_debug"
        save_csv(df_submission, filename, folder)
        # Save in leaderboard format
        filename = run_id + "_" + timestr + "leaderboard"
        save_csv(df_submission[["pair_id", "generated_question"]], filename, folder)

    # Save generation parameters in a json file for reference
    save_params(args, filename, folder)


def keep_best_bleurt(df):
    # Keep only the question with the best BLEURT score
    return df.sort_values("bleurt_score", ascending=False).drop_duplicates(["pair_id"])


def compute_overall_bleurt_score(df_pred):
    df_pred = keep_best_bleurt(df_pred)
    print("Mean BLEURT grouped by question attribute type:\n", df_pred.groupby('attribute1')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question local vs summary:\n", df_pred.groupby('local_or_sum')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT grouped by question explicit vs implicit:\n", df_pred.groupby('ex_or_im')['bleurt_score'].agg(['mean', 'count']))
    print("Mean BLEURT over all samples: ", df_pred['bleurt_score'].mean())


def pool_first_10_ranking(df):
    # Keep first 10 questions generated for each pair id
    df = df.groupby("pair_id").apply(lambda x: x.head(10))

    return df


def baseline_bleurt_first_10_questions(args, run_id):
    # Load test set
    dir_name = os.path.join(RAW_DIR, args.eval_folder)
    df_test = load_df(args.eval_filename, dir_name)

    # Get bleurt score without ranking = pick first 10 questions for each pair id
    if( args.eval_type == "local_val"):
        df_no_ranking = pool_first_10_ranking(df_test)
        save_submission(df_no_ranking, args, run_id)
        compute_overall_bleurt_score(df_no_ranking)


def main():
    args = add_params()
    # Get saved models dir
    if ( torch.cuda.is_available() ):
        saved_models_dir = "/work/nigel_umass_edu/qg_challenge/saved_models/"
    else:
        saved_models_dir = "../saved_models/"
    run_id = args.model_folder.split("/")[1]
    args.model_folder = os.path.join(saved_models_dir, args.model_folder)
    # Set random seed if specified
    if args.seed != -1:
        set_random_seed(args.seed)
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'No gpu found!'

    # Run score (bleurt) prediction model for evaluation
    predictions, df_test = predict_bleurt(args, device)
    
    # Get top-10 generated questions according to bleurt score prediction for each pair id
    df_submission = pool(predictions, df_test)
    # Save top-10 generated questions for each pair id in submission format
    save_submission(df_submission, args, run_id)
    # Get overall bleurt score using pooled top-10 questions
    if( args.eval_type == "local_val"):
        compute_overall_bleurt_score(df_submission)

    
if __name__ == '__main__':
    main()