import os
import pandas as pd
from create_dataset_split import load_dfs_v2, save_csv


#%% Load file
RAW_DIR = "/home/zw16/Quest_Question_Generation_Challenge/data"
eval_folder = 'testset'
eval_filename = 'multiStrategy_umass_cur_best_02052023_0.90_1.00_16'
folder = os.path.join(RAW_DIR, "{}/results".format(eval_folder))
df_pred_raw = load_dfs_v2(eval_filename, folder)#, nrows=10)

#%% rerank and select top 10 by ppl
## for each pair_id group, rerank by the ppl score, 
## then select the top 10, and then concatenate the resulting groups
df_pred_sorted = []
for pair_id, group in df_pred_raw.groupby('pair_id'):
    group_sorted = group.sort_values(by=['ppl'], ascending=True)
    # deduplicate
    group_sorted = group_sorted.drop_duplicates(subset=['generated_question'])
    if len(group_sorted) < 10:
        df_pred_sorted.append(group_sorted)
    else:
        df_pred_sorted.append(group_sorted.head(10))
df_pred = pd.concat(df_pred_sorted)

#%% Get only pair_id and generated_question, then save
df_pred = df_pred[['pair_id', 'generated_question']]
save_csv(df_pred, 
         "{}_ppl_reranked".format(eval_filename), folder)