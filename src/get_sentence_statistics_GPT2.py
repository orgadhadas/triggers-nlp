import json
from collections import defaultdict

import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
import argparse

from Utils import get_sentences_from_file, get_sentence_loss, parse_args


def parse_args():
    parser = argparse.ArgumentParser(description='Get sentence statistics (mean log proba, std) for each length.')
    parser.add_argument('--input_path', '-i', required=True, type=str, help='input path')
    parser.add_argument('--output_path', '-o', required=True, type=str, help='save path')
    # parser.add_argument('--task', required=True, type=str, help="task to run stats on", choices=['snli', 'sst', 'squad'])
    # parser.add_argument('--label', required=True, type=str, help="label to take subset of dataset by")
    # parser.add_argument('--trigger_length', required=True, type=int, help="Which triggered dataset to take (can be 0 for no trigger)")
    args = parser.parse_args()
    return args


def main(args):
    sentences = get_sentences_from_file(args.input_path)
    random.shuffle(sentences)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")

    d = defaultdict(list)

    for s in tqdm(sentences):
        loss = get_sentence_loss(model, tokenizer, s)
        d[len(tokenizer(s)['input_ids'])].append(loss)

    output_dict = {"length": [], "mean": [], "std": [], "N": []}
    for length in d.keys():
        output_dict[length] = {"mean": np.mean(d[length]), "std": np.std(d[length]), "N": len(d[length])}
        # output_dict["length"].append(length)
        # output_dict["mean"].append(np.mean(d[length]))
        # output_dict["std"].append(np.std(d[length]))
        # output_dict["N"].append(len(d[length]))

    with open(args.output_path, 'w') as json_file:
        json.dump(output_dict, json_file, sort_keys=True)



if __name__ == '__main__':
    args = parse_args()
    main(args)