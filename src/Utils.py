import argparse
import torch
import os
import numpy as np

def parse_args_old():
    parser = argparse.ArgumentParser(description='Get sentence statistics (mean log proba, std) for each length.')
    parser.add_argument('--path', '-p', required=True, type=str, help='save path')
    args = parser.parse_args()
    return args

def clean_spaces(s : str):
    return s.replace(" .", ".").replace(" ,", ",").replace(" '", "'").replace(" !", "!").replace(" ?", "?")\
        .replace(" :", ":").replace(" ;", ";")


def get_clean_sentences_from_file(file_path):
    with open(file_path) as f:
        sentences = f.readlines()
    sentences = list(map(lambda x: x[:-10], sentences))
    sentences = list(map(lambda x: clean_spaces(x), sentences))
    return sentences


def get_sentence_log_proba(model, tokenizer, s):
    sentence = tokenizer.bos_token + s
    input = tokenizer(sentence, return_tensors='pt')
    res = model(**input)
    return res.logits[0, range(input['input_ids'].size(1)-1), input['input_ids'][0][1:]].sum().item()

def get_sentence_loss(model, tokenizer, s):
    input = tokenizer.encode(s, return_tensors='pt')
    with torch.no_grad():
        model.eval()
        loss = model(input.to(model.device), labels=input.to(model.device)).loss
    return loss.item()


def get_tuples_trigger_file_path_num_trigger_list(dir_path):
    '''

    :param dir_path: path to the directory where all trigger files are.
    :return: list of tuples from the shape: (trigger_file_path, num_of_triggered_word)
    '''
    tuple_list = list()
    for triggered_file_name in os.listdir(dir_path):
        trigger = triggered_file_name.split(".")[-1]
        trigger_ends_location = len(trigger.split('_'))
        trigger_file_path = dir_path + '/' + triggered_file_name
        tuple_list.append((trigger_file_path, trigger_ends_location))
    return tuple_list


def parse_args():
    parser = argparse.ArgumentParser(description='Get model result.')
    parser.add_argument('--dir', '-d', required=True, type=str, help='The folder of the triggered files')
    args = parser.parse_args()
    return args

def get_array_mean_std(l: list):
    arr = np.array(l)
    return arr.mean(), arr.std()

# def get_mean_std_from_rate_list(rate_per_len_dict):
