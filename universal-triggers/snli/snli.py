import argparse
import os
import sys

import wandb as wandb
from sklearn.neighbors import KDTree
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
    parser.add_argument("--src", help="Subset of examples to attack (which label)", type=str, choices=['entailment', 'contradiction', 'neutral'])
    parser.add_argument("--dst", help="target label to change the classifier to", type=str, choices=['entailment', 'contradiction', 'neutral'])
    return parser.parse_args()

def init_wandb(args):
    wandb.init(project="triggers-nlp",
               config={
                   "task": "snli",
                   "source_label": args.src,
                   "destination_label": args.dst
               })

def main():
    args = parse_args()
    init_wandb(args)
    # Load SNLI dataset
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    # Load model and vocab
    model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    model.eval().cuda()
    vocab = model.vocab

    # add hooks for embeddings so we can compute gradients w.r.t. to the input tokens
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

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
    for instance in dev_dataset:
        if instance['label'].label == dataset_label_filter:
            subset_dev_dataset.append(instance)
    # the attack is targeted towards a specific class
    target_label_dic = {
        'entailment': "0",
        'contradiction': "1",
        'neutral': "2"
    }
    target_label = target_label_dic[args.dst]
    # target_label = "0" # flip to entailment
    # target_label = "1" # flip to contradiction
    # target_label = "2" # flip to neutral

    # A k-d tree if you want to do gradient + nearest neighbors
    #tree = KDTree(embedding_weight.numpy())

    all_triggers = []

    # Get original accuracy before adding universal triggers
    _, acc = utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids=None, snli=True)
    all_triggers.append(["", acc])
    model.train() # rnn cannot do backwards in train mode

    # Initialize triggers
    num_trigger_tokens = 1 # one token prepended
    trigger_token_ids = [vocab.get_token_index("a")] * num_trigger_tokens
    # sample batches, update the triggers, and repeat
    tmp = 0
    for batch in lazy_groups_of(iterator(subset_dev_dataset, num_epochs=10, shuffle=True), group_size=1):
        # get model accuracy with current triggers
        trigger, acc = utils.get_accuracy(model, subset_dev_dataset, vocab, trigger_token_ids, snli=True)
        all_triggers.append([trigger[:-1], acc])
        model.train() # rnn cannot do backwards in train mode

        # get grad of triggers
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids, target_label, snli=True)

        # find attack candidates using an attack method
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=40)
        # cand_trigger_token_ids = attacks.random_attack(embedding_weight,
        #                                                trigger_token_ids,
        #                                                num_candidates=40)
        # cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
        #                                                        embedding_weight,
        #                                                        trigger_token_ids,
        #                                                        tree,
        #                                                        100,
        #                                                        decrease_prob=True)

        # query the model to get the best candidates
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids,
                                                      snli=True)
        # tmp+=1
        # if tmp > 10:
        #     break

    # triggers_table = wandb.Table(data=all_triggers, columns=["trigger", "accuracy"])
    # wandb.run.log({"triggers_table": triggers_table})
    pd.DataFrame(all_triggers, columns=["trigger", "accuracy"]).to_csv('triggers.csv', index=False)
    wandb.save(f'triggers{args.src}_{args.dst}.csv')
    # os.remove('triggers.csv')

if __name__ == '__main__':
    main()
