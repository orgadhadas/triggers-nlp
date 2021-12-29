import pickle
from collections import defaultdict, Counter
import argparse
import json
import torchtext as tt
import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
from Utils import get_sentences_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Get model result.')
    parser.add_argument('--triggered', '-t', required=True, type=str, help='The file with triggeres')
    parser.add_argument('--clean', '-c', required=True, type=str, help='The file with orignal sentences (no trigger)')
    parser.add_argument('--thr', required=True, type=float, help='Threshold for bi-gram')
    parser.add_argument('--train', required=False, default=None, type=str, help='If you want to train a new bi-gram model, input the file to train from')
    parser.add_argument('--bigram', required=False, default=None, type=str, help='Path to bi-gram json file')
    args = parser.parse_args()
    return args

def init_wandb(args):
    wandb.init(project="triggers-detection-test", config={
        "model": "n-gram",
        "threshold": args.thr,
        "triggered_file": args.triggered,
        "clean_file": args.clean,
        "n": 2,
        "train_data": args.train,
        "model_path": args.bigram,
        "with_eos": True
    })

def train_bi_gram_model(input_path, output_path):
    '''
    :param input_path: Input path to file that we read training sentences
    :param output_path: Output path to dump the dictionary
    :return:
    '''

    print("Training a bi-gram model using file " + input_path)

    bi_gram_dict = defaultdict(lambda: defaultdict(int))
    counter_first_word = Counter()
    sentences = get_sentences_from_file(input_path)
    tokenizer = tt.data.get_tokenizer("basic_english")

    for s in tqdm(sentences):
        tokens_list = tokenizer(s)

        # TMP - but maybe make this permanent
        tokens_list.append("#eos#")
        tokens_list.insert(0, "#eos#")

        for (w1, w2) in zip(tokens_list[:-1], tokens_list[1:]):
            counter_first_word[w1] += 1
            bi_gram_dict[w1][w2] += 1

    for w1 in list(bi_gram_dict.keys()):
        for w2 in list(bi_gram_dict[w1].keys()):
            bi_gram_dict[w1][w2] = bi_gram_dict[w1][w2] / counter_first_word[w1]

    import pickle
    with open(output_path, 'wb') as handle:
        pickle.dump(dict(bi_gram_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bi_gram_dict


def test(path, b_dict, thr):
    """
    :param path: Path to take sentences from
    :param b_dict: The bigram dictionary.
    :param gt_trigger_ends_location: Where triggers tokens ends!
    :return:
    num_attacked: The number of the sentences that model recognized that consist trigger.
    num_recognized_trigger_location: The number of the trigger locations that were find correctly.
    num_sentences: The number of sentences in the file.
    """

    num_attack_detected = 0
    tokenizer = tt.data.get_tokenizer("basic_english")
    data = pd.read_csv(path, header=None)

    num_recognized_trigger_location = 0
    for _, (sentence, trigger) in data.iterrows():

        gt_trigger_ends_location = len(tokenizer(trigger)) + 1 # Index of the last word in the trigger + 1
        tokens_list = tokenizer(sentence)

        # TMP - but maybe make this permanent
        tokens_list.append("#eos#")
        tokens_list.insert(0, "#eos#")

        attacked = 0
        trigger_location = 0
        for i, (w1, w2) in enumerate(zip(tokens_list[:-1], tokens_list[1:])):
            if b_dict[w1][w2] <= thr:
                trigger_location += 1
                attacked = 1
            else:
                break

        num_attack_detected += attacked
        # print(trigger_location, gt_trigger_ends_location)
        if attacked == 0:
            for i, (w1, w2) in enumerate(zip(tokens_list[:-1], tokens_list[1:])):
                print(w1, w2, b_dict[w1][w2])
        if trigger_location == gt_trigger_ends_location:
            num_recognized_trigger_location += 1

    return num_attack_detected, num_recognized_trigger_location, len(data)

def test_not_triggered(path, b_dict, thr):
    """
    :param path: Path to take sentences from
    :param b_dict: The bigram dictionary.
    :param gt_trigger_ends_location: Where triggers tokens ends!
    :return:
    num_attacked: The number of the sentences that model recognized that consist trigger.
    num_recognized_trigger_location: The number of the trigger locations that were find correctly.
    num_sentences: The number of sentences in the file.
    """

    num_attack_detected = 0
    tokenizer = tt.data.get_tokenizer("basic_english")

    sentences = get_sentences_from_file(path)

    for sentence in sentences:

        tokens_list = tokenizer(sentence)

        # TMP - but maybe make this permanent
        tokens_list.append("#eos#")
        tokens_list.insert(0, "#eos#")

        attacked = 0
        trigger_location = 0
        for i, (w1, w2) in enumerate(zip(tokens_list[:-1], tokens_list[1:])):
            # print(w1, w2, b_dict[w1][w2])
            if b_dict[w1][w2] <= thr:
                trigger_location += 1
                attacked = 1
            else:
                break

        num_attack_detected += attacked

    return num_attack_detected, 0, len(sentences)


def main(args):

    init_wandb(args)

    if args.train is not None:
        b_dict = train_bi_gram_model(args.train, args.bigram)
    else:
        with open(args.bigram, 'rb') as handle:
            b_dict = pickle.load(handle)
            b_dict = defaultdict(lambda: defaultdict(int), b_dict)
        print(f"Loaded bi-gram model from {args.bigram}")

    tp_trigger_presence, num_correct_trigger_location, num_sentences = test(args.triggered, b_dict, args.thr)

    acc = num_correct_trigger_location / num_sentences
    tpr_trigger_presence = tp_trigger_presence / num_sentences

    # Note that positive is triggered sentence. So false positive is the number of sentences that model think they are
    # triggered. We know that fp_trigger_presence is false because the input is clean entailment (no trigger at the
    # beginning).
    fp_trigger_presence, _, num_false_sentences = test_not_triggered(args.clean, b_dict, args.thr)
    FPR = fp_trigger_presence / num_false_sentences

    print(f"bigram FPR: {FPR}")
    print(f"bigram TPR: {tpr_trigger_presence}")
    print(f"bigram ACC: {acc}")

    wandb.summary["FPR"] = FPR
    wandb.summary["ACC"] = acc
    wandb.summary["TPR"] = tpr_trigger_presence

if __name__ == '__main__':
    args = parse_args()
    main(args)