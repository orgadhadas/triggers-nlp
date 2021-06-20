from collections import defaultdict, Counter
from utils import clean_spaces
import json
import torchtext as tt
from tqdm import tqdm
import os
import numpy as np


def train_bi_gram_model(input_path, output_path):
    '''

    :param input_path: Input path to file that we read training sentences
    :param output_path: Output path to dump the dictionary
    :return:
    '''
    bi_gram_dict = defaultdict(lambda: defaultdict(int))
    counter_first_word = Counter()
    with open(input_path) as f:
        sentences = f.readlines()
    sentences_new = list(map(lambda x: x[:-10], sentences))
    sentences_new = list(map(lambda x: clean_spaces(x), sentences_new))

    for s in tqdm(sentences_new):
        tokens_list = tt.data.get_tokenizer("basic_english")(s)
        # tokens_list = ['<s>'] + tokens_list

        for (w1, w2) in zip(tokens_list[:-1], tokens_list[1:]):
            counter_first_word[w1] += 1
            bi_gram_dict[w1][w2] += 1

    for w1 in list(bi_gram_dict.keys()):
        for w2 in list(bi_gram_dict[w1].keys()):
            bi_gram_dict[w1][w2] = bi_gram_dict[w1][w2] / counter_first_word[w1]

    with open(output_path, 'w') as json_file:
        json.dump(bi_gram_dict, json_file, indent=4)
    return bi_gram_dict


def test(path, b_dict, add_start_token=False, gt_trigger_ends_location=0):
    """

    :param path: Path to take sentences from
    :param b_dict: The bigram dictionary.
    :param add_start_token: Whatever to add the beginning sentence token.
    todo: add_start_token might be irrelevant since we use only False
    :param gt_trigger_ends_location: Where triggers tokens ends!
    todo: what if trigger token is more than 1? that mean one word more than one token? how to check it?
    :return:
    num_attacked: The number of the sentences that model recognized that consist trigger.
    num_recognized_trigger_location: The number of the trigger locations that were find correctly.
    num_sentences: The number of sentences in the file.
    """

    num_attacked = 0
    tokenizer = tt.data.get_tokenizer("basic_english")
    with open(path) as f:
        sentences = f.readlines()
    if gt_trigger_ends_location != -1:
        triggered_word = sentences[0].split()[:gt_trigger_ends_location]
        triggered_word = " ".join(triggered_word)
        gt_trigger_ends_location = len(tokenizer(triggered_word))

    num_recognized_trigger_location = 0
    for s in sentences:
        tokens = tokenizer(s)
        if add_start_token:
            tokens_list = ['<s>'] + tokens
        else:
            tokens_list = tokens
        attacked = 0
        trigger_location = 0
        for i, (w1, w2) in enumerate(zip(tokens_list[:-1], tokens_list[1:])):
            if b_dict[w1][w2] == 0:
                trigger_location += 1
                attacked = 1
            else:
                break

        num_attacked += attacked
        if trigger_location == gt_trigger_ends_location:
            num_recognized_trigger_location += 1

    # percent_trigger_presence = num_attacked/len(sentences)
    # print(f'% of attacked sentences {percent_trigger_presence}')
    # percent_correct_trigger_location = num_recognized_trigger_location/len(sentences)
    # print(f'% of sentences that trigger location recognized {percent_correct_trigger_location}')
    return num_attacked, num_recognized_trigger_location, len(sentences)


def main():
    add_start_token = False
    b_dict = train_bi_gram_model(r'../data/train_data_label_entailment_uniq.txt', r'../data/bi_gram_entailment_dict.json')
    triggers_path = r'../data/triggered_data'
    entailment_clean_path = r'../data/dev_data_label_entailment_uniq.txt'
    trigger_max_len = 5
    trigger_stats = [[] for _ in range(trigger_max_len)]
    tp_trigger_presence, total_num_correct_trigger_location, total_num_sentences = 0, 0, 0 # todo understand..
    for triggered_file_name in os.listdir(triggers_path):
        trigger = triggered_file_name.split(".")[-1]
        trigger_ends_location = len(trigger.split('_'))
        trigger_path = triggers_path + '/' + triggered_file_name
        num_trigger_presence, num_correct_trigger_location, num_sentences = test(trigger_path, b_dict, add_start_token=add_start_token, gt_trigger_ends_location=trigger_ends_location)
        tp_trigger_presence += num_trigger_presence
        total_num_correct_trigger_location += num_correct_trigger_location
        total_num_sentences += num_sentences
        trigger_stats[trigger_ends_location - 1].append(num_correct_trigger_location / num_sentences)
    print("table 3 result \n bigram")
    for trigger_len in range(trigger_max_len):
        mean_acc = np.array(trigger_stats[trigger_len-1]).mean()
        std_acc = np.array(trigger_stats[trigger_len-1]).std()
        print(f"trigger len: {trigger_len + 1} trigger mean acc: {mean_acc} +- {std_acc}")


    recall_trigger_presence = tp_trigger_presence / total_num_sentences
    print(f'trigger location acc: {total_num_correct_trigger_location/total_num_sentences}')
    print("entailment")
    fp_trigger_presence, _, num_false_sentences = test(entailment_clean_path, b_dict,
                                                       add_start_token=add_start_token)
    print(f"bigram FPR: {fp_trigger_presence/num_false_sentences}")


    # precision_trigger_presence = tp_trigger_presence / (tp_trigger_presence + fp_trigger_presence)
    #
    # f1_trigger_presence = 2 / (1 / precision_trigger_presence + 1 / recall_trigger_presence)
    # print(f"precision trigger presence: {precision_trigger_presence} recall trigger presence: {recall_trigger_presence} "
    #       f"f1 trigger presence: {f1_trigger_presence}")


if __name__ == '__main__':
    main()