import argparse
import json
import torch
import wandb
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm
from collections import defaultdict

from Utils import *


def check_if_trigger_exist(ex, tokenizer, model, train_stats, num_tokens_in_ex):
    '''

    :param ex: The example sentence
    :param tokenizer: Tokenizer for the model
    :param model:  GPT model that we calculate it's loss
    :param train_stats: Statistics of the loss on the training set.
    :param num_tokens_in_ex:  The number of tokens in the input sentence.
    :return: True if we detected trigger False otherwise.
    '''
    # return 1 if there is a trigger at the beginning.
    loss = get_sentence_loss(model, tokenizer, ex)
    return loss > train_stats[str(num_tokens_in_ex)]['mean'] + train_stats[str(num_tokens_in_ex)]['std']


def find_trigger_location(ex, tokenizer, model, train_stats, num_tokens_in_ex):
    '''

    :param ex: The sentence
    :param tokenizer: Tokenizer for the model
    :param model: GPT model that we use in order to find trigger location.
    :param train_stats: Statistics of the loss on the training set.
    :param num_tokens_in_ex: The number of tokens in the input sentence.
    :return: The location of the trigger in the sentence. If return 0 that means no trigger. If -1 there is not enough
    statistics for this sentence.
    '''
    trigger_length_found = 0
    try:
        while check_if_trigger_exist(ex, tokenizer, model, train_stats, num_tokens_in_ex):
            trigger_length_found += 1
            ex = ex.split()[1:]
            ex = " ".join(w for w in ex)
            num_tokens_in_ex = len(tokenizer(ex)['input_ids'])
    except:
        print("missing stats length in find trigger loaction")
        return -1
    return trigger_length_found


def get_recognition_stats(tokenizer, model, triggers_folder, data_orig, train_stats):
    '''

    :param tokenizer: Tokenizer for the model
    :param model: GPT model that we find recognition stats with.
    :param triggers_folder: Folder where the filer with the trigger
    :param data_orig: The dataset without the trigger
    :param train_stats: Statistics of the loss on the training set.
    :return:
    '''
    fp = 0
    tp = 0
    failure_in_find_trigger_location = 0

    wandb.init(project="triggers-detection-test", config={
        "model": "GPT2",
        "threshold": "varying according to length. mean(loss) + std(loss)",
    })


    total_correct_num_tested_location = 0
    total_num_triggered_sentences = 0
    tp_rec_per_len_dict = defaultdict(list)
    acc_per_len_dict = defaultdict(list)

    all_triggers = []
    for triggered_file_path, trigger_length, trigger_name in get_tuples_trigger_file_path_num_trigger_list(triggers_folder):
        all_triggers.append(trigger_name)
        sentences = get_clean_sentences_from_file(triggered_file_path)
        num_trigger_presence = 0
        num_correct_trigger_location = 0
        total_num_triggered_sentences += len(sentences)
        for ex in tqdm(sentences):
            num_tokens_in_ex = len(tokenizer(ex)['input_ids'])
            try:
                if check_if_trigger_exist(ex, tokenizer, model, train_stats, num_tokens_in_ex):
                    num_trigger_presence += 1
                trigger_length_found = find_trigger_location(ex, tokenizer, model, train_stats, num_tokens_in_ex)
                if trigger_length_found != -1:
                    if trigger_length_found == trigger_length:
                        num_correct_trigger_location += 1
                else:
                    failure_in_find_trigger_location += 1
            except:
                failure_in_find_trigger_location += 1
                print(f"No train example with length {num_tokens_in_ex}, skipping...")
                continue
        tp += num_trigger_presence
        total_correct_num_tested_location += num_correct_trigger_location
        acc_per_len_dict[trigger_length].append(num_correct_trigger_location / len(sentences))
        tp_rec_per_len_dict[trigger_length].append(num_trigger_presence / len(sentences))

    acc_mean_per_len = [0] * len(acc_per_len_dict)
    acc_std_per_len = [0] * len(acc_per_len_dict)
    tpr_mean_per_len = [0] * len(acc_per_len_dict)
    tpr_std_per_len = [0] * len(acc_per_len_dict)

    for trigger_len in tp_rec_per_len_dict.keys():
        mean_tp_rec, std_tp_rec = get_array_mean_std(tp_rec_per_len_dict[trigger_len])
        mean_acc_location, std_acc_location = get_array_mean_std(acc_per_len_dict[trigger_len])
        print(f"trigger len: {trigger_len} trigger location mean acc: {mean_acc_location} +- {std_acc_location} "
              f"trigger recognize mean TPR: {mean_tp_rec} +- {std_tp_rec}")
        acc_mean_per_len[trigger_len-1] = mean_acc_location
        acc_std_per_len[trigger_len-1] = std_acc_location
        tpr_mean_per_len[trigger_len-1] = mean_tp_rec
        tpr_std_per_len[trigger_len-1] = std_tp_rec
        wandb.log({"Mean_acc_per_length": mean_acc_location, "Mean_tpr_per_length": mean_tp_rec, "len": trigger_len})


    print(f"GPT2 TPR: {tp / total_num_triggered_sentences}")
    for ex in tqdm(data_orig):
        num_tokens_in_ex = len(tokenizer(ex)['input_ids'])
        try:
            if check_if_trigger_exist(ex, tokenizer, model, train_stats, num_tokens_in_ex):
                fp += 1
        except:
            print(f"No train example with length {str(num_tokens_in_ex)}, skipping...")
            continue
    print(f'trigger location acc: {total_correct_num_tested_location / total_num_triggered_sentences}')
    print(f"number of failure in find trigger location due missing statistics {failure_in_find_trigger_location}")
    print(f"% of failure in find trigger location due missing statistics {failure_in_find_trigger_location / total_num_triggered_sentences}")
    print(f'trigger location acc if we consider only example that did not fail due missing statistics'
          f': {total_correct_num_tested_location / (total_num_triggered_sentences- failure_in_find_trigger_location)}')

    FPR = fp / len(data_orig)
    print(f"GPT2 FPR: {FPR}")

    wandb.summary["FPR"] = FPR
    wandb.summary["ACC_per_length_mean"] = acc_mean_per_len
    wandb.summary["ACC_per_length_std"] = acc_std_per_len
    wandb.summary["ACC_mean_overall"] = np.mean(acc_mean_per_len)
    wandb.summary["ACC_std_overall"] = np.std(acc_mean_per_len)
    wandb.summary["TPR_per_length_mean"] = tpr_mean_per_len
    wandb.summary["TPR_per_length_std"] = tpr_std_per_len
    wandb.summary["TPR_mean_overall"] = np.mean(tpr_mean_per_len)
    wandb.summary["TPR_std_overall"] = np.std(tpr_mean_per_len)
    wandb.summary["inspected_triggers"] = all_triggers



def main(args):
    with open(args.stats) as f:
      train_stats = json.load(f)

    triggers_folder = args.dir
    data_orig = get_clean_sentences_from_file(args.clean)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelWithLMHead.from_pretrained("gpt2").to(device)
    get_recognition_stats(tokenizer, model, triggers_folder, data_orig, train_stats)


if __name__ == '__main__':
    args = parse_args_with_stats()
    main(args)
