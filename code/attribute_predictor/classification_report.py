import os
import pandas as pd 
from sklearn.metrics import classification_report

file_name = 'attr_flan_t5_large_val.csv'
file_path = os.path.join('./data/results/train_val_split_csv', file_name)
df = pd.read_csv(file_path)

def preprocess_pred_attr(pred_attrs):
    prefix = 'The attribute is: '
    clean_attrs = []
    for pred_attr in pred_attrs:
        clean_attrs.append(pred_attr.split(prefix)[1])
    return clean_attrs

clean_pred_attrs = preprocess_pred_attr(df['predicted attribute'].tolist())
true_attrs = df['attribute1'].tolist()
print(classification_report(true_attrs, clean_pred_attrs))





