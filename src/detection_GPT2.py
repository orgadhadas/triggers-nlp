import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm
from collections import defaultdict

from Utils import *


def get_args():
    parser = argparse.ArgumentParser(description='Detection code for triggers based on GPT-2 statistics.')
    parser.add_argument('--dir', required=True, type=str, help='path to the folder with all triggered files')
    parser.add_argument('--path_no_triggers', required=True, type=str, help='file path without triggers')
    parser.add_argument('--path_statistics', required=True, type=str, help='file of train statistics')
    args = parser.parse_args()
    return args


def check_if_trigger_exist(ex, tokenizer, model, train_stats, length):
    # return 1 if there is a trigger at the beginning.
    loss = get_sentence_loss(model, tokenizer, ex)
    return loss > train_stats[str(length)]['mean'] + train_stats[str(length)]['std']


def get_recognition_stats(tokenizer, model, triggers_folder, data_orig, train_stats):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    correct = 0
    total_num_triggered_sentences = 0
    tp_rec_per_len_dict = defaultdict(list)

    for triggered_file_path, trigger_length in get_tuples_trigger_file_path_num_trigger_list(triggers_folder):
        sentences = get_clean_sentences_from_file(triggered_file_path)
        total_num_triggered_sentences += len(sentences)
        num_trigger_presence = 0
        for ex in tqdm(sentences):
            length = len(tokenizer(ex)['input_ids'])
            try:
                if check_if_trigger_exist(ex, tokenizer, model, train_stats, length):
                    num_trigger_presence += 1
                    correct += 1
                else:
                    fn += 1
            except:
                print(f"No train example with length {length}, skipping...")
                continue
        tp += num_trigger_presence
        tp_rec_per_len_dict[trigger_length].append(num_trigger_presence / len(sentences))

    for trigger_len in tp_rec_per_len_dict.keys():
        mean_tp_rec, std_tp_rec = get_array_mean_std(tp_rec_per_len_dict[trigger_len])
        print(f"trigger len: {trigger_len}  "
              f"trigger recognize mean TPR: {mean_tp_rec} +- {std_tp_rec}")


    print(f"GPT2 TPR: {tp / total_num_triggered_sentences}")
    for ex in tqdm(data_orig):
        length = len(tokenizer(ex)['input_ids'])
        try:
            if check_if_trigger_exist(ex, tokenizer, model, train_stats, length):
                fp += 1
            else:
                tn += 1
                correct += 1
        except:
            print(f"No train example with length {str(length)}, skipping...")
            continue

    print(f"GPT2 FPR: {fp / len(data_orig)}")
    num_total_sentences = total_num_triggered_sentences + len(data_orig)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = correct / num_total_sentences
    f1 = (precision * recall) / (precision + recall)

    # print(f"Recall: {recall}, Precision: {precision}, F1: {f1}, Accuracy: {acc}")


def main(args):
    with open(args.path_statistics) as f:
      train_stats = json.load(f)

    triggers_folder = args.dir
    # data_triggers = data_triggers[:100]
    data_orig = get_clean_sentences_from_file(args.path_no_triggers)
    # data_orig = data_orig[:100]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"running on device: {device}")
    model = AutoModelWithLMHead.from_pretrained("gpt2").to(device)
    # model = AutoModelWithLMHead.from_pretrained("gpt2")
    get_recognition_stats(tokenizer, model, triggers_folder, data_orig, train_stats)


if __name__ == '__main__':
    args = get_args()
    main(args)
