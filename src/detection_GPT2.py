import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

from Utils import clean_spaces, get_sentence_loss

parser = argparse.ArgumentParser(description='Detection code for triggers based on GPT-2 statistics.')
parser.add_argument('--path_triggers', required=True, type=str, help='file path with triggers')
parser.add_argument('--path_no_triggers', required=True, type=str, help='file path without triggers')
parser.add_argument('--path_statistics', required=True, type=str, help='file of train statistics')
args = parser.parse_args()

with open(args.path_statistics) as f:
  train_stats = json.load(f)

with open(args.path_triggers) as f:
    data_triggers = f.readlines()

with open(args.path_no_triggers) as f:
    data_orig = f.readlines()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

fp = 0
tp = 0
fn = 0
tn = 0
correct = 0
total = 0

data_triggers = list(map(lambda x : x[:-10], data_triggers))
data_triggers = list(map(lambda x : clean_spaces(x), data_triggers))
#data_triggers = data_triggers[:100]

data_orig = list(map(lambda x : x[:-10], data_orig))
data_orig = list(map(lambda x : clean_spaces(x), data_orig))
#data_orig = data_orig[:100]

for ex in tqdm(data_triggers):
    loss = get_sentence_loss(model, tokenizer, ex)
    length = len(tokenizer(ex)['input_ids'])
    if str(length) not in train_stats:
        print(f"No train example with length {str(length)}, skipping...")
        continue
    if loss > train_stats[str(length)]['mean'] + train_stats[str(length)]['std']:
        # mark as trigger
        tp += 1
        correct += 1
    else:
        fn += 1

print("% of triggered examples detected: ", tp / len(data_triggers))

for ex in tqdm(data_orig):
    loss = get_sentence_loss(model, tokenizer, ex)
    length = len(tokenizer(ex)['input_ids'])
    if str(length) not in train_stats:
        print(f"No train example with length {str(length)}, skipping...")
        continue
    if loss > train_stats[str(length)]['mean'] + train_stats[str(length)]['std']:
        # mark as trigger
        fp += 1
    else:
        tn += 1
        correct += 1

print("% of real examples detected: ", fp / len(data_orig))

total = len(data_triggers) + len(data_orig)

recall = tp / (tp + fn)
precision = tp / (tp + fp)
acc = correct / total
f1 = (precision * recall) / (precision + recall)

print(f"Recall: {recall}, Precision: {precision}, F1: {f1}, Accuracy: {acc}")