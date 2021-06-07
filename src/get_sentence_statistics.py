import argparse
from collections import defaultdict

import tqdm as tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np
import json
from tqdm import tqdm
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Get sentence statistics (mean log proba, std) for each length.')
    parser.add_argument('--path', '-p', required=True, type=str, help='save path')
    args = parser.parse_args()
    return args

def clean_spaces(s : str):
    return s.replace(" .", ".").replace(" ,", ",").replace(" '", "'").replace(" !", "!").replace(" ?", "?")\
        .replace(" :", ":").replace(" ;", ";")

def get_sentence_log_proba(model, tokenizer, s):
    sentence = tokenizer.bos_token + s
    input = tokenizer(sentence, return_tensors='pt')
    res = model(**input)
    return res.logits[0, range(input['input_ids'].size(1)-1), input['input_ids'][0][1:]].sum().item()

def get_sentence_loss(model, tokenizer, s):
    input = tokenizer.encode(s, return_tensors='pt')
    with torch.no_grad():
        model.eval()
        loss = model(input, labels=input).loss
    return loss.item()





# with open("../data/train_data_label_contradiction_uniq.txt") as f:
#     contradiction = f.readlines()

with open("../data/train_data_label_entailment_uniq.txt") as f:
    entailment = f.readlines()

# with open("../data/train_data_label_neutral_uniq.txt") as f:
#     neutral = f.readlines()



sentences = entailment
# sentences = []
# base_dir = "../data"
# prefix = "dev_data_label_entailment_uniq.txt."
# trigger_list = ["spacecraft_goals_pager_drowning_facial", "xbox_masons_cyclone_clusters_netball", "no_scarlet_shred_cloned_wildebeest", 
# "astronauts_slumbers_investment_sherpa_hourly", "beef_tiered_inning_canoeing_yello"]
# for trigger in trigger_list:
#     with open(os.path.join(base_dir, (prefix + trigger))) as f:
#         sentences += f.readlines()




sentences_new = list(map(lambda x : x[:-10], sentences))
sentences_new = list(map(lambda x : clean_spaces(x), sentences_new))
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

d = defaultdict(list)

# d = defaultdict(list)

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
