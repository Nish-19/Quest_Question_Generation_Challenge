import os
from code.utils.create_dataset_split import load_df, save_csv

RAW_DIR = "./data"


def drop_duplicate_samples(df):
        # Count no of duplicate answers
        print("No of samples before deduplication: ", len(df))
        total_duplicates = len(df[df.duplicated(subset=['pair_id', 'generated_question'], keep=False)])
        unique_duplicates = df.groupby(['pair_id', 'generated_question']).size().gt(1).sum()
        print(f"No of duplicate samples removed = {total_duplicates - unique_duplicates} = {round((total_duplicates - unique_duplicates)/len(df) * 100, 2)}%")
        # Remove duplicates where both answer and question are the same
        df = df.drop_duplicates(subset=["pair_id", "generated_question"], keep="first")

        return df


folder = os.path.join(RAW_DIR, f"augmentation/flan_t5_nischal/")
# Combine all parsed files into a single file
filename = "nucleus_flan_t5_large_0.95_1.20.csv"
df = load_df(filename, folder)
df = drop_duplicate_samples(df)
print("No of samples after deduplication: ", len(df))


folder = os.path.join(RAW_DIR, f"augmentation/flan_t5_nischal/")
filename = "nucleus_flan_t5_large_0.95_1.20_no_duplicates"
save_csv(df.reset_index(drop=True), filename, folder)