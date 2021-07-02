import pickle
from collections import defaultdict

import torch
import wandb as wandb
from scipy.stats import hmean
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM

from Utils import parse_args, get_tuples_trigger_file_path_num_trigger_list, get_clean_sentences_from_file, \
    parse_args_with_stats


def get_hmean_score(example, tokenizer, model, device):
    '''

    :param example: The sentence example
    :param tokenizer: Tokenizer for the model
    :param model: Bert like Model
    :param device: GPU or CPU
    :return: harmonic mean score of probas
    '''
    first_word = example.split()[0]
    tokens = tokenizer.encode(first_word, add_special_tokens=False)
    length = len(tokens)
    replacement = "<mask> " * length
    example_masked = example.replace(first_word, replacement[:-1])
    with torch.no_grad():
        model.eval()
        probas = model(tokenizer.encode(example_masked, return_tensors='pt').to(device)).logits.softmax(dim=2)[
            0, range(1, length + 1), tokens].cpu().detach().numpy()
    hmean_score = hmean(probas)
    return hmean_score


def get_statistics(sentences, tokenizer, model, device):
    '''

    :param sentences: sentences to calculate the statistics
    :param tokenizer: Tokenizer for the model
    :param model: Bert like Model
    :param device: GPU or CPU
    :return: mean and std of all sentences score
    '''

    all_hmean = []
    for s in tqdm(sentences):
        hmean = get_hmean_score(s, tokenizer, model, device)
        all_hmean.append(hmean)

    return np.mean(all_hmean), np.std(all_hmean)


def detect_trigger(example, tokenizer, model, thr, device):
    '''

    :param example: Sentence to detect trigger.
    :param tokenizer: Tokenizer for the model
    :param model: Bert like Model
    :param thr: Threshold for the hmean score. Above this threshold we consider the sentence as True
    :param device: GPU or CPU
    :return: True if we detected trigger False otherwise.
    '''

    score1 = get_hmean_score(example, tokenizer, model, device)

    if score1 < thr:
        return True
    else:
        return False


def detect_location_of_trigger(example, tokenizer, model, thr, device):
    '''

    :param example: The sentence that we want to find trigger location.
    :param tokenizer: Tokenizer for the model
    :param model: Bert like Model
    :param thr: Threshold for the hmean score. Above this threshold we consider the sentence as True
    :param device: GPU or CPU
    :return: The location of the trigger in the sentences -1 if we detected that there is no trigger.
    '''

    for i in range(len(example)):
        if not detect_trigger(example.split(' ', i)[-1], tokenizer, model, thr, device):
            return i - 1

    # if all stats are super small, then we assume no trigger
    return -1


def get_mean_std_lists(d):
    mean_d = [0] * len(d.items())
    std_d = [0] * len(d.items())
    for k,v in d.items():
        mean_d[k-1] = np.mean(v)
        std_d[k-1] = np.std(v)
    return mean_d, std_d


def log_items(acc_list, tpr_list):
    for k in range(len(acc_list)):
        wandb.log({"Mean_acc_per_length": acc_list[k], "Mean_tpr_per_length": tpr_list[k], "len": k+1})


def __main__(args):
    stats = pickle.load(open(args.stats, "rb"))
    mean_score = stats['mean']
    std_score = stats['std']
    thr = mean_score + 5 * std_score

    wandb.init(project="triggers-detection-test", config={
        "model": "roBERTa",
        "threshold": thr,
    })

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    device = torch.device("cuda")
    model.to(device)

    acc_per_len = defaultdict(list)
    tpr_per_len = defaultdict(list)
    all_acc = []
    all_tpr = []
    all_triggers = []

    for triggered_file_path, trigger_length, trigger_name in get_tuples_trigger_file_path_num_trigger_list(args.dir):
        data_triggers = get_clean_sentences_from_file(triggered_file_path)

        n_correct = 0
        tp = 0
        fn = 0

        print(trigger_length, trigger_name)
        for ex in tqdm(data_triggers):
            loc = detect_location_of_trigger(ex, tokenizer, model, thr, device)
            if loc == trigger_length-1:
                n_correct += 1
            if loc >= 0:
                tp += 1
            else:
                fn += 1

        acc = n_correct / (tp + fn)
        TPR = tp / (tp + fn)
        print("ACC", acc, "TPR", TPR)
        acc_per_len[trigger_length].append(acc)
        all_acc.append(acc)
        tpr_per_len[trigger_length].append(TPR)
        all_tpr.append(TPR)
        all_triggers.append(trigger_name)

    fp = 0
    tn = 0
    data_orig = get_clean_sentences_from_file(args.clean)
    for ex in tqdm(data_orig):
        if detect_trigger(ex, tokenizer, model, thr, device):
            fp += 1
        else:
            tn += 1
    FPR = fp / (tn + fp)

    acc_mean_per_len, acc_std_per_len = get_mean_std_lists(acc_per_len)
    tpr_mean_per_len, tpr_std_per_len = get_mean_std_lists(tpr_per_len)
    acc_mean_overall = np.mean(all_acc)
    acc_std_overall = np.std(all_acc)
    tpr_mean_overall = np.mean(all_tpr)
    tpr_std_overall = np.std(all_tpr)

    log_items(acc_mean_per_len, tpr_mean_per_len)


    wandb.summary["FPR"] = FPR
    wandb.summary["ACC_per_length_mean"] = acc_mean_per_len
    wandb.summary["ACC_per_length_std"] = acc_std_per_len
    wandb.summary["ACC_mean_overall"] = acc_mean_overall
    wandb.summary["ACC_std_overall"] = acc_std_overall
    wandb.summary["TPR_per_length_mean"] = tpr_mean_per_len
    wandb.summary["TPR_per_length_std"] = tpr_std_per_len
    wandb.summary["TPR_mean_overall"] = tpr_mean_overall
    wandb.summary["TPR_std_overall"] = tpr_std_overall
    wandb.summary["inspected_triggers"] = all_triggers

if __name__ == '__main__':
    args = parse_args_with_stats()
    __main__(args)
