from collections import defaultdict, Counter
import argparse
import json
import torchtext as tt
from tqdm import tqdm
import numpy as np

from src.Utils import get_clean_sentences_from_file, get_tuples_trigger_file_path_num_trigger_list, parse_args


def train_bi_gram_model(input_path, output_path):
    '''

    :param input_path: Input path to file that we read training sentences
    :param output_path: Output path to dump the dictionary
    :return:
    '''
    bi_gram_dict = defaultdict(lambda: defaultdict(int))
    counter_first_word = Counter()
    sentences = get_clean_sentences_from_file(input_path)

    for s in tqdm(sentences):
        tokens_list = tt.data.get_tokenizer("basic_english")(s)

        for (w1, w2) in zip(tokens_list[:-1], tokens_list[1:]):
            counter_first_word[w1] += 1
            bi_gram_dict[w1][w2] += 1

    for w1 in list(bi_gram_dict.keys()):
        for w2 in list(bi_gram_dict[w1].keys()):
            bi_gram_dict[w1][w2] = bi_gram_dict[w1][w2] / counter_first_word[w1]

    with open(output_path, 'w') as json_file:
        json.dump(bi_gram_dict, json_file, indent=4)
    return bi_gram_dict


def test(path, b_dict, gt_trigger_ends_location=0):
    """

    :param path: Path to take sentences from
    :param b_dict: The bigram dictionary.
    :param gt_trigger_ends_location: Where triggers tokens ends!
    :return:
    num_attacked: The number of the sentences that model recognized that consist trigger.
    num_recognized_trigger_location: The number of the trigger locations that were find correctly.
    num_sentences: The number of sentences in the file.
    """

    num_attacked = 0
    tokenizer = tt.data.get_tokenizer("basic_english")
    sentences = get_clean_sentences_from_file(path)
    if gt_trigger_ends_location != -1:
        triggered_word = sentences[0].split()[:gt_trigger_ends_location]
        triggered_word = " ".join(triggered_word)
        gt_trigger_ends_location = len(tokenizer(triggered_word))

    num_recognized_trigger_location = 0
    for s in sentences:
        tokens_list = tokenizer(s)
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

    return num_attacked, num_recognized_trigger_location, len(sentences)


def main(triggers_dir_path):
    b_dict = train_bi_gram_model(r'../data/train_data_label_entailment_uniq.txt', r'../data/bi_gram_entailment_dict.json')
    # triggers_dir_path = r'../data/triggered_data'
    entailment_clean_path = r'../data/dev_data_label_entailment_uniq.txt'
    acc_per_len_dict = defaultdict(list)
    tp_rec_per_len_dict = defaultdict(list)

    tp_trigger_presence, total_num_correct_trigger_location, total_num_sentences = 0, 0, 0

    for triggered_file_path, trigger_ends_location in get_tuples_trigger_file_path_num_trigger_list(triggers_dir_path):
        num_trigger_presence, num_correct_trigger_location, num_sentences = test(triggered_file_path, b_dict, gt_trigger_ends_location=trigger_ends_location)
        tp_trigger_presence += num_trigger_presence
        total_num_correct_trigger_location += num_correct_trigger_location
        total_num_sentences += num_sentences
        acc_per_len_dict[trigger_ends_location].append(num_correct_trigger_location / num_sentences)
        tp_rec_per_len_dict[trigger_ends_location].append(num_trigger_presence / num_sentences)
    print("bigram")
    print(f"acc over all triggered sentences: {total_num_correct_trigger_location/total_num_sentences}")
    mean_location_acc_dict = dict()
    std_location_acc_dict = dict()
    mean_tp_rec_dict = dict()
    std_tp_rec_dict = dict()

    for trigger_len in acc_per_len_dict.keys():
        mean_acc_location = np.array(acc_per_len_dict[trigger_len]).mean()
        mean_location_acc_dict[trigger_len] = mean_acc_location
        std_acc_location = np.array(acc_per_len_dict[trigger_len]).std()
        std_location_acc_dict[trigger_len] = std_acc_location
        mean_tp_rec = np.array(tp_rec_per_len_dict[trigger_len]).mean()
        mean_tp_rec_dict[trigger_len] = mean_tp_rec
        std_tp_rec = np.array(tp_rec_per_len_dict[trigger_len]).std()
        std_tp_rec_dict[trigger_len] = np.array(std_tp_rec)
        print(f"trigger len: {trigger_len} trigger location mean acc: {mean_acc_location} +- {std_acc_location} "
              f"trigger recognize mean TPR: {mean_tp_rec} +- {std_tp_rec}")


    tpr_trigger_presence = tp_trigger_presence / total_num_sentences
    print(F"bigram TPR: {tpr_trigger_presence}")
    print(f'trigger location acc: {total_num_correct_trigger_location/total_num_sentences}')

    # Note that positive is triggered sentence. So false positive is the number of sentences that model think they are
    # triggered. We know that fp_trigger_presence is false because the input is clean entailment (no trigger at the
    # beginning).
    fp_trigger_presence, _, num_false_sentences = test(entailment_clean_path, b_dict)
    print(f"bigram FPR: {fp_trigger_presence/num_false_sentences}")




if __name__ == '__main__':
    args = parse_args()
    main(args.dir)