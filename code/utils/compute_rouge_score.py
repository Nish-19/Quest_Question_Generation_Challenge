import os
import statistics
import argparse
from rouge_score import rouge_scorer
from code.utils.create_dataset_split import load_df, RAW_DIR

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-EFD', "--eval_folder", type=str, default="train_val_split_csv", help="Folder containing evaluation file relative to data folder")
    parser.add_argument('-EFN', "--eval_filename", type=str, default="val.csv", help="Evaluation filename with timestamp and .csv extension")

    params = parser.parse_args()
    
    return params

def main():
    args = add_params()
    folder = os.path.join(RAW_DIR, "results_org")
    pred_df = load_df(args.eval_filename, folder)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores_pre, scores_rec, scores_f1 = [], [], []
    for i, row in pred_df.iterrows():
        score = scorer.score(row['question'], row['generated_question'])
        scores_pre.append(score['rougeL'].precision)
        scores_rec.append(score['rougeL'].recall)
        scores_f1.append(score['rougeL'].fmeasure)
    
    mean_rl_pre = statistics.mean(scores_pre)
    mean_rl_rec = statistics.mean(scores_rec)
    mean_rl_f1 = statistics.mean(scores_f1)

    print('#### Mean Rouge-L Scores ####')
    print('Precision: {:.4f}'.format(mean_rl_pre))
    print('Recall: {:.4f}'.format(mean_rl_rec))
    print('F1 score: {:.4f}'.format(mean_rl_f1))

    
if __name__ == '__main__':
    main()