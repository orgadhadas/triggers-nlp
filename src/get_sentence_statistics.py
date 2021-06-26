import json
from collections import defaultdict

import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
import argparse

from Utils import get_clean_sentences_from_file, get_sentence_loss, parse_args


def get_parsed_input_path():
    parser = argparse.ArgumentParser(description='Get sentence statistics (mean log proba, std) for each length.')
    parser.add_argument('--input_path', '-i', required=True, type=str, help='input path')
    parser.add_argument('--output_path', '-o', required=True, type=str, help='save path')
    args = parser.parse_args()
    return args


def main(args):
	sentences = get_clean_sentences_from_file(args.input_path)
	random.shuffle(sentences)
	tokenizer = AutoTokenizer.from_pretrained("gpt2")
	model = AutoModelWithLMHead.from_pretrained("gpt2")

	d = defaultdict(list)

	for s in tqdm(sentences):
	    # log_proba = get_sentence_log_proba(model, tokenizer, s)
	    loss = get_sentence_loss(model, tokenizer, s)
	    # d[len(tokenizer(s)['input_ids'])].append(log_proba)
	    d[len(tokenizer(s)['input_ids'])].append(loss)


	output_dict = {}
	for length in d.keys():
	    output_dict[length] = {"mean": np.mean(d[length]), "std": np.std(d[length]), "N": len(d[length])}

	with open(args.output_path, 'w') as json_file:
	    json.dump(output_dict, json_file, sort_keys=True)


if __name__ == '__main__':
    args = get_parsed_input_path()
    main(args)