import argparse
import os
import sys

import wandb as wandb
# from sklearn.neighbors import KDTree
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.iterators import BasicIterator
sys.path.append('..')
import utils
import attacks
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Compute triggers for MNLI.')
    parser.add_argument("--src", help="Subset of examples to attack (which label)", type=str, choices=['entailment', 'contradiction', 'neutral'], required=True)
    parser.add_argument("--dst", help="target label to change the classifier to", type=str, choices=['entailment', 'contradiction', 'neutral'], required=True)
    parser.add_argument("--trigger", help="the trigger to test on", type=str, required=True)
    parser.add_argument("--model", help="pretrained model to attacks", type=str, choices=['ESIM', 'DA', 'combined', 'DA-ELMO'], required=True)
    return parser.parse_args()

# def init_wandb(args):
#     wandb.init(project="triggers-nlp",
#                config={
#                    "task": "snli",
#                    "source_label": args.src,
#                    "destination_label": args.dst,
#                    "length": args.len,
#                    "model": args.model
#                })

def main():
    args = parse_args()
    # init_wandb(args)
    # Load SNLI dataset
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    # train_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
    # Load model and vocab
    if args.model == 'ESIM':
        model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    if args.model == 'DA':
        model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model
    if args.model == 'combined':
        model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
        model2 = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz').model
        model2.eval().cuda()
    if args.model == 'DA-ELMO':
        model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz').model

    model.eval().cuda()
    vocab = model.vocab

    # add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    if args.model == 'combined':
        utils.add_hooks(model2)

    embedding_weight = utils.get_embedding_weight(model, args.model) # save the word embedding matrix

    # Batches of examples to construct triggers
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Subsample the dataset to one class to do a universal attack on that class
    dataset_label_filter = args.src
    # dataset_label_filter = 'entailment' # only entailment examples
    # dataset_label_filter = 'contradiction' # only contradiction examples
    # dataset_label_filter = 'neutral' # only neutral examples
    subset_dev_dataset = []
    subset_train_dataset = []
    for instance in dev_dataset:
        if instance['label'].label == dataset_label_filter:
            subset_dev_dataset.append(instance)
    # for instance in train_dataset:
    #     if instance['label'].label == dataset_label_filter:
    #         subset_train_dataset.append(instance)
    # the attack is targeted towards a specific class
    target_label_dic = {
        'entailment': 0,
        'contradiction': 1,
        'neutral': 2
    }
    target_label = target_label_dic[args.dst]

    trigger_token_ids = []
    for token in args.trigger.split(" "):
        trigger_token_ids.append(vocab.get_token_index(token))

    # Get original accuracy before adding universal triggers
    _, acc = utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids=trigger_token_ids, snli=True, target_label=target_label)

if __name__ == '__main__':
    main()
