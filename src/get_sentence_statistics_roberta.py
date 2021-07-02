import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
from transformers import RobertaForMaskedLM, RobertaTokenizer

from Utils import get_clean_sentences_from_file
from roberta import get_hmean_score

parser = argparse.ArgumentParser(description='Get statistics for roberta.')
parser.add_argument('--file', '-f', required=True, type=str, help='The file with to get statistics on')
args = parser.parse_args()


def get_statistics(sentences, tokenizer, model, device):
    '''

    :param sentences: sentences for statistic
    :param tokenizer: Tokenizer for the model
    :param model: The a Bert model we calculate
    :param device: GPU or CPU
    :return:
    '''
    all_hmean = []
    for s in tqdm(sentences):
        hmean = get_hmean_score(s, tokenizer, model, device)
        all_hmean.append(hmean)

    return np.mean(all_hmean), np.std(all_hmean)


sentences = get_clean_sentences_from_file(args.file)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
device = torch.device("cuda")
model.to(torch.device("cuda"))
mean, std = get_statistics(sentences, tokenizer, model, device)
d = {"mean": mean, "std": std}

with open('roberta_stats.pickle', 'wb') as f:
    pickle.dump(d, f)
