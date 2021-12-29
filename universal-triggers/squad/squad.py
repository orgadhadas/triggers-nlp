import argparse
import sys

import wandb
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.iterators import BasicIterator
sys.path.append('..')
import utils
import squad_utils
import attacks
import pandas as pd

def init_wandb(args):
    wandb.init(project="triggers-nlp",
               config={
                   "task": "squad",
                   "source_label": args.question,
                   "destination_label": args.target,
                   "length": 6,
                   "model": "BIDAF"
               })

def parse_args():
    parser = argparse.ArgumentParser(description='Compute triggers for MNLI.')
    parser.add_argument("--model", help="pretrained model to attacks", type=str, choices=['BIDAF'], required=True)
    parser.add_argument("--target", help="the answer we want to get", type=str, required=True)
    # parser.add_argument("--template", help="the template for trigger, out of choices", type=int, required=True, choices=[0,1,2,3])
    parser.add_argument("--question", help="target answer", type=str, required=True,
                        choices=['who', 'what', 'where', 'when', 'how', 'why', 'which'])
    return parser.parse_args()

def main():
    args = parse_args()
    init_wandb(args)
    # Read the SQuAD validation dataset using a word tokenizer
    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SquadReader(token_indexers={'tokens': single_id})
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json')
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

    # Use batches to craft the universal perturbations
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # We register a gradient hook on the embeddings.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

    # Initialize the trigger. The first one is an intialization with all "the" tokens.
    # You can customize it. Make sure to set the fixed target answer and the question type.
    # The second is a trigger found after running as reported in our paper.

    template_dic = {
        "why": "the the the the the the {0} the",
        "who": "the the the the {0} the the the the",
        "when": "the the the the {0} the the the the",
        "where": "the the the the {0} the the the the",
    }

    trigger_init = template_dic[args.question].format(args.target)
    target_answer = args.target

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

    # trigger_init = "why how ; known because : to kill american people ."
    # target_answer = "to kill american people"
    # subsampled_dev_dataset = why_questions_dev # universal attack on `who` questions

    # tokenizes the trigger, and finds the start/end span
    # make sure the trigger tokens are space separated
    trigger_token_ids = [vocab.get_token_index(t) for t in trigger_init.split(' ')]
    span_start = trigger_init.split(' ').index(target_answer.split(' ')[0]) # start of target_answer
    span_end = trigger_init.split(' ').index(target_answer.split(' ')[-1])
    # we ignore replacement at the positions of the answer (answer is fixed)
    ignore_indices = [0]*(span_start) + \
        [1]*(span_end - span_start + 1) + [0]*(len(trigger_token_ids) - 1 - span_end)

    # if these parameters are bigger = better result, but slower
    num_candidates = 20
    beam_size = 5

    all_triggers = []
    for _ in range(100):
        # Get targeted accuracy
        f1, em, trigger_string = squad_utils.get_accuracy_squad(model,
                                       subsampled_dev_dataset,
                                       vocab,
                                       trigger_token_ids,
                                       target_answer,
                                       span_start,
                                       span_end)
        model.train()
        all_triggers.append([trigger_string, f1, em])

        # Get the gradient for the appended tokens averaged over the batch.
        averaged_grad = squad_utils.get_average_grad_squad(model,
                                                           vocab,
                                                           trigger_token_ids,
                                                           subsampled_dev_dataset,
                                                           span_start,
                                                           span_end)

        # Use an attack method to get the top candidates
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=num_candidates,
                                                        increase_loss=False)

        # Query the model with the top candidates to find the best tokens.
        trigger_token_ids = squad_utils.get_best_candidates_squad(model, trigger_token_ids,
                                                                  cand_trigger_token_ids,
                                                                  vocab,
                                                                  subsampled_dev_dataset,
                                                                  beam_size,
                                                                  ignore_indices,
                                                                  span_start,
                                                                  span_end)

    # for final trigger
    f1, em, trigger_string = squad_utils.get_accuracy_squad(model,
                                                            subsampled_dev_dataset,
                                                            vocab,
                                                            trigger_token_ids,
                                                            target_answer,
                                                            span_start,
                                                            span_end)
    all_triggers.append([trigger_string, f1, em])

    file_name = f"triggers_{args.question}_{args.target}.csv"
    pd.DataFrame(all_triggers, columns=["trigger", "f1", "em"]).to_csv(file_name, index=False)
    wandb.save(file_name)

if __name__ == '__main__':
    main()
