import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Get sentence statistics (mean log proba, std) for each length.')
    parser.add_argument('--path', '-p', required=True, type=str, help='save path')
    args = parser.parse_args()
    return args

def clean_spaces(s : str):
    return s.replace(" .", ".").replace(" ,", ",").replace(" '", "'").replace(" !", "!").replace(" ?", "?")\
        .replace(" :", ":").replace(" ;", ";")

def get_sentence_log_proba(model, tokenizer, s):
    sentence = tokenizer.bos_token + s
    input = tokenizer(sentence, return_tensors='pt')
    res = model(**input)
    return res.logits[0, range(input['input_ids'].size(1)-1), input['input_ids'][0][1:]].sum().item()

def get_sentence_loss(model, tokenizer, s):
    input = tokenizer.encode(s, return_tensors='pt')
    with torch.no_grad():
        model.eval()
        loss = model(input, labels=input).loss
    return loss.item()
