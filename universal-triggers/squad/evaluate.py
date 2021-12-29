import argparse
import sys

import wandb
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
sys.path.append('..')
import utils
import squad_utils
import attacks

def parse_args():
    parser = argparse.ArgumentParser(description='Compute triggers for MNLI.')
    parser.add_argument("--model", help="pretrained model to attacks", type=str, choices=['BIDAF'], required=True)
    parser.add_argument("--trigger", help="pretrained model to attacks", type=str, required=True)
    parser.add_argument("--target", help="target answer", type=str, required=True)
    parser.add_argument("--question", help="target answer", type=str, required=True, choices=['who', 'what', 'where', 'when', 'how', 'why', 'which'])
    return parser.parse_args()

# def init_wandb(args):
#     wandb.init(project="triggers-nlp-test",
#                config={
#                    "task": "squad",
#                    "trigger": args.trigger,
#                    "target": args.target,
#                    "source": args.question,
#                    "model": args.model
#                })

def main():
    args = parse_args()
    # init_wandb(args)
    # Read the SQuAD validation dataset using a word tokenizer
    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SquadReader(token_indexers={'tokens': single_id})
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json')
    # Load the model and its associated vocabulary.
    model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-glove-2019.05.09.tar.gz').model
    vocab = model.vocab
    model.eval().cuda()

    # filter to just certain `wh` questions
    who_questions_dev, what_questions_dev, where_questions_dev, when_questions_dev, what_questions_dev, \
        how_questions_dev, why_questions_dev, which_questions_dev, other_questions_dev = ([] for i in range(9))
    for item in dev_dataset:
        for word in item['question']:
            if word.text.lower() == 'who':
                who_questions_dev.append(item)
                break
            if word.text.lower() == 'what':
                what_questions_dev.append(item)
                break
            if word.text.lower() == 'where':
                where_questions_dev.append(item)
                break
            if word.text.lower() == 'when':
                when_questions_dev.append(item)
                break
            if word.text.lower() == 'how':
                how_questions_dev.append(item)
                break
            if word.text.lower() == 'why':
                why_questions_dev.append(item)
                break
            if word.text.lower() == 'which':
                which_questions_dev.append(item)
                break
            else:
                other_questions_dev.append(item)

    trigger_init = args.trigger
    target_answer = args.target

    subsampled_dev_dataset = None
    if args.question == 'where':
        subsampled_dev_dataset = where_questions_dev
    if args.question == 'why':
        subsampled_dev_dataset = why_questions_dev
    if args.question == 'who':
        subsampled_dev_dataset = who_questions_dev
    if args.question == 'how':
        subsampled_dev_dataset = how_questions_dev
    if args.question == 'what':
        subsampled_dev_dataset = what_questions_dev
    if args.question == 'when':
        subsampled_dev_dataset = when_questions_dev
    if args.question == 'which':
        subsampled_dev_dataset = which_questions_dev

    # make sure the trigger tokens are space separated
    trigger_token_ids = [vocab.get_token_index(t) for t in trigger_init.split(' ')]
    span_start = trigger_init.split(' ').index(target_answer.split(' ')[0]) # start of target_answer
    span_end = trigger_init.split(' ').index(target_answer.split(' ')[-1])

    squad_utils.get_accuracy_squad(model,
                                   subsampled_dev_dataset,
                                   vocab,
                                   trigger_token_ids,
                                   target_answer,
                                   span_start,
                                   span_end)

if __name__ == '__main__':
    main()