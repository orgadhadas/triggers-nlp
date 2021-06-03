import sys
from allennlp.data.tokenizers import WordTokenizer    
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.models import load_archive
from allennlp.data.token_indexers import SingleIdTokenIndexer

def main():
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences
    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    train_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
    model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    # model.eval().cuda()
    vocab = model.vocab

    label = 'contradiction'
    f_without_trigger = open(f"train_data_label_{label}.txt", "w+")
    # f_with_trigger = open("triggered_dev_data.txt", "a+")
    # trigger_str = ''
    # for i, idx in enumerate(trigger_token_ids):
    #     if i == 0:
    #         trigger_str = vocab.get_token_from_index(idx)
    #     else:
    #         trigger_str += (' ' + vocab.get_token_from_index(idx))

    for d in train_dataset:
        sentence = list(d['premise'])
        if d['label'].label == label:

            # print(d['label'].label)
            sentecne_to_no_trigger = ''
            
            for i, word in enumerate(sentence):
               
                if i == 0:
                   sentecne_to_no_trigger = str(word)
                else:
                    sentecne_to_no_trigger += (' ' + str(word))
            # sentence_with_trigger = trigger_str + ' ' + sentecne_to_no_trigger
            # sentence_with_trigger = sentecne_to_no_trigger + ' ' + trigger_str
            f_without_trigger.write(sentecne_to_no_trigger + '\n')
            # f_with_trigger.write(sentence_with_trigger + '\n')

    f_without_trigger.close()
        



if __name__ == '__main__':
    main()
