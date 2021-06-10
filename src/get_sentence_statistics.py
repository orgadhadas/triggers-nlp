import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from Utils import clean_spaces, get_sentence_loss, parse_args

with open("../data/train_data_label_entailment_uniq.txt") as f:
    entailment = f.readlines()

# with open("../data/train_data_label_neutral_uniq.txt") as f:
#     neutral = f.readlines()

# with open("../data/train_data_label_contradiction_uniq.txt") as f:
#     contradiction = f.readlines()


sentences = entailment

sentences_new = list(map(lambda x : x[:-10], sentences))
sentences_new = list(map(lambda x : clean_spaces(x), sentences_new))
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

d = defaultdict(list)

for s in tqdm(sentences_new):
    # log_proba = get_sentence_log_proba(model, tokenizer, s)
    loss = get_sentence_loss(model, tokenizer, s)
    # d[len(tokenizer(s)['input_ids'])].append(log_proba)
    d[len(tokenizer(s)['input_ids'])].append(loss)


output_dict = {}
for length in d.keys():
    output_dict[length] = {"mean": np.mean(d[length]), "std": np.std(d[length]), "N": len(d[length])}

args = parse_args()
with open(args.path, 'w') as json_file:
    json.dump(output_dict, json_file, sort_keys=True)
