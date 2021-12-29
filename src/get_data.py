import sys
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.models import load_archive
from allennlp.data.token_indexers import SingleIdTokenIndexer
from argparse import ArgumentParser
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from tqdm import tqdm
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from Utils import clean_spaces


class SnliData:
    def __init__(self, args_obj, single_id_indexer, tokenizer):
        reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
        test_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl')
        dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
        train_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
        if args_obj.snli_all:
            self.labels_list = ['contradiction', 'entailment', 'neutral']
        else:
            labels_list = list()
            if args_obj.contradiction:
                labels_list.append('contradiction')
            if args_obj.entailment:
                labels_list.append('entailment')
            if args_obj.neutral:
                labels_list.append('neutral')
            self.labels_list = labels_list
        self.data_set_list = [(train_dataset, 'train_data'), (dev_dataset, 'dev_data'), (test_dataset, 'test_data')]
        self.key_in_data = 'premise'
        self.key_for_label = 'label'


class SstData:
    def __init__(self, args_obj, single_id_indexer):
        # reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
        #                                                 token_indexers={"tokens": single_id_indexer},
        #                                                 use_subtrees=True) # this is the reader to the train
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                        token_indexers={"tokens": single_id_indexer})

        train_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')

        dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

        if args_obj.sst_all:
            self.labels_list = ['0', '1']
        else:
            labels_list = list()
            if args_obj.zero:
                labels_list.append('0')
            if args_obj.one:
                labels_list.append('1')
            self.labels_list = labels_list
        self.data_set_list = [(train_dataset, 'train_data'), (dev_dataset, 'dev_data')]
        self.key_in_data = 'tokens'
        self.key_for_label = 'label'


class SQuADData:
    def __init__(self, args_obj, single_id):
        reader = SquadReader(token_indexers={'tokens': single_id})
        # they used only in the dev.
        dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json')
        train_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json')
        if args_obj.squad_all:
            self.labels_list = ['who', 'what', 'where', 'when', 'how', 'why', 'which']
        else:
            labels_list = list()
            if args_obj.who:
                labels_list.append('who')
            if args_obj.what:
                labels_list.append('what')
            if args_obj.where:
                labels_list.append('where')
            if args_obj.when:
                labels_list.append('when')
            if args_obj.how:
                labels_list.append('how')
            if args_obj.why:
                labels_list.append('why')
            if args_obj.which:
                labels_list.append('which')

            self.labels_list = labels_list
        self.data_set_list = [(dev_dataset, 'dev_data'), (train_dataset, 'train_data')]
        self.key_in_data = 'passage'
        self.key_for_label = 'question'


def main():
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True)  # word tokenizer
    # tokenizer = WordTokenizer(end_tokens=["@@NULL@@"])  # add @@NULL@@ to the end of sentences
    tokenizer = WordTokenizer(end_tokens=[""])
    task_list = []
    if args.snli:
        snli_data = SnliData(args, single_id_indexer, tokenizer)
        task_list.append((snli_data, 'snli'))

    if args.sst:
        sst_data = SstData(args, single_id_indexer)
        task_list.append((sst_data, 'sst'))

    if args.squad:
        squad_data = SQuADData(args, 'squad')
        task_list.append((squad_data, 'squad'))

    files_dict = dict()
    for task_data, task_name in task_list:
        files_dict[task_name] = dict()
        for (_, data_set_name) in task_data.data_set_list:
            files_dict[task_name][data_set_name] = dict()
            for label in task_data.labels_list:
                files_dict[task_name][data_set_name][label] = open(f"../data/{task_name}_{data_set_name}_label_{label}.txt", "w+")

    for task_data, task_name in task_list:
        for (data, data_set_name) in task_data.data_set_list:
            for d in tqdm(data):
                current_data = list(d[task_data.key_in_data])
                if task_name in {'snli', 'sst'}:
                    label = d[task_data.key_for_label].label  # todo.label might not exist in other data!!!!!
                else:
                    label = d[task_data.key_for_label][0].text.lower()
                if label in task_data.labels_list:
                    sentence_to_no_trigger = ''
                    for i, word in enumerate(current_data):
                        if i == 0:
                            sentence_to_no_trigger = str(word)
                        else:
                            sentence_to_no_trigger += (' ' + str(word))
                    files_dict[task_name][data_set_name][label].write(clean_spaces(sentence_to_no_trigger) + '\n')

    for task_data, task_name in task_list:
        for (_, data_set_name) in task_data.data_set_list:
            for label in task_data.labels_list:
                files_dict[task_name][data_set_name][label].close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--snli", dest="snli", action='store_true', required=False, default=False)
    parser.add_argument("--snli_all", dest='snli_all', action='store_true', required=False, default=False)
    parser.add_argument("--contradiction", dest='contradiction', action='store_true', required=False, default=False)
    parser.add_argument("--entailment", dest='entailment', action='store_true', required=False, default=False)
    parser.add_argument("--neutral", dest='neutral', action='store_true', required=False, default=False)
    parser.add_argument("--sst", dest="sst", action='store_true', required=False, default=False)
    parser.add_argument("--sst_all", dest='sst_all', action='store_true', required=False, default=False)
    parser.add_argument("--0", dest='zero', action='store_true', required=False, default=False)
    parser.add_argument("--1", dest='one', action='store_true', required=False, default=False)

    parser.add_argument("--squad", dest="squad", action='store_true', required=False, default=False)
    parser.add_argument("--who", dest='who', action='store_true', required=False, default=False)
    parser.add_argument("--where", dest='where', action='store_true', required=False, default=False)
    parser.add_argument("--when", dest='when', action='store_true', required=False, default=False)
    parser.add_argument("--how", dest="how", action='store_true', required=False, default=False)
    parser.add_argument("--why", dest='why', action='store_true', required=False, default=False)
    parser.add_argument("--what", dest='what', action='store_true', required=False, default=False)
    parser.add_argument("--which", dest='which', action='store_true', required=False, default=False)
    parser.add_argument("--squad_all", dest='squad_all', action='store_true', required=False, default=False)

    args = parser.parse_args()
    main()
