import evaluate
import string
import argparse
import re
import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## number of generation for each answer is 100
NUM_GEN = 100


file = pd.read_csv('flan_t5_large_100_bluert.csv')
context = [prompt[55: prompt.find(' The answer is ')] for prompt in file['prompt']]
answer = [prompt[prompt.find(' The answer is ')+15:] for prompt in file['prompt']]
bleurt = np.array(file['bleurt_score'])

binary_bleurt = [1 if i >= 0.7 else 0 for i in bleurt]

bleurt_order = []
for i in range(0, len(bleurt), NUM_GEN):
    bleurt_order += [sorted(bleurt[i:i+NUM_GEN], reverse=True).index(val) for val in bleurt[i:i+NUM_GEN]]

file['context'] = context
file['answer'] = answer
file['bleurt_order'] = bleurt_order
file['binary_bleurt'] = binary_bleurt

sentences = []
for idx, row in file.iterrows():
    sentences.append('The answer is ' + row['answer'] + ' The generated question is ' + row['generated_question'])


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
encoded_input = tokenizer(sentences, padding=True, return_tensors='pt')


def train_test_split(X, mask, y, num_gen=NUM_GEN, train_size=0.8, seed=42):
    random.seed(seed)
    num_ques = len(X) // num_gen
    train_index = random.sample(range(num_ques), int(train_size*num_ques))
    test_index  = np.setdiff1d(range(num_ques), train_index)
    train_idx, test_idx = [], []
    for i in train_index:
        train_idx += range(i*num_gen, (i+1)*num_gen)
    for i in test_index:
        test_idx += range(i*num_gen, (i+1)*num_gen)
    X_train, y_train, = X[train_idx], y[train_idx]
    mask_train, mask_test = mask[train_idx], mask[test_idx]
    X_test, y_test  = X[test_idx], y[test_idx]
    return X_train, y_train, mask_train, mask_test, X_test, y_test, train_index, test_index

ids_num = len(file['pair_id'].unique())
X_train, y_train, mask_train, mask_test, X_test, y_test, train_index, test_index = train_test_split(encoded_input['input_ids'],
                                                                                 encoded_input['attention_mask'], 
                                                                                 np.array(binary_bleurt),
                                                                                 seed = 0)

batch_size = 8

def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader   


train_dataloader = create_dataloaders(X_train, mask_train, y_train, batch_size)
test_dataloader = create_dataloaders(X_test, mask_test, y_test, batch_size)


class Regressor(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(Regressor, self).__init__()
        D_in, D_out = 768, 1
        self.lm = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        # self.lm = AutoModel.from_pretrained("roberta-base")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        
    def forward(self, input_ids, attention_masks):
        outputs = self.lm(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs
    
    
model = Regressor(drop_rate=0.2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

loss_function = nn.MSELoss()

def evaluate_test(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss = []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels.float())

        test_loss.append(loss.item())
    test_loss = sum(test_loss) / len(test_loss)
    return test_loss

def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, test_dataloader, device, clip_value=2, print_every=1000):
    for epoch in range(epochs):
        print("-----")
        best_loss = 1e10
        model.train()
        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs, batch_labels.float())
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step() 
            if (step + 1) % print_every == 0:
                print(f'Epoch {epoch}, Step {step+1} with training loss of {loss}')
                test_loss = evaluate_test(model, loss_function, test_dataloader, device)
                print(f'Epoch {epoch}, Step {step+1} with testing loss of {test_loss}')
                print('=' * 50)
    return model

model = train(model, optimizer, scheduler, loss_function, epochs, 
              train_dataloader, test_dataloader, device, clip_value=2)

def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks).view(1,-1).tolist()[0]
    return output

y_pred_scaled = predict(model, test_dataloader, device)

def evaluate(num_test, prediction, y_test):
    max_bleurt = []
    max_pred_bleurt = []
    top_10_pred = []
    for idx in range(num_test):
        pred = prediction[idx*100: (idx+1)*100]
        true = y_test[idx*100: (idx+1)*100]
        max_pred_bleurt.append(true[np.argmax(pred)])
        max_bleurt.append(max(true))
        
        top_10_idx = np.argsort(pred)[-10:]
        top_10_values = true[top_10_idx]
        top_10_pred.append(max(top_10_values))
    
    print(f'TOP 1 BLEURT in prediction is {np.average(max_pred_bleurt)}, \
            TOP 10 BLEURT is {np.average(top_10_pred)}')
    print(f'Best true BLEURT is {np.average(max_bleurt)}')
        
    return max_pred_bleurt, max_bleurt

_, _ = evaluate(len(test_index), y_pred_scaled, y_test)

torch.save(model, 'model/regressor')
