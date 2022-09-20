# from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer
from tcn_test_4_1_2.data_tcn.MyDataset import MyDataset
from tcn_test_4_1_2.data_tcn.parameters import Parameters
from torchtext.vocab import build_vocab_from_iterator


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
parameters = Parameters()


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['[PAD]', '[CLS]', '[SEP]'])
    vocab.set_default_index(vocab["[PAD]"])
    return vocab


def data_iter(dataset: MyDataset):
    for index in range(0, dataset.__len__()):
        yield dataset.__getitem__(index)


def yield_tokens(d_iter):
    for _, text in d_iter:
        if type(text) == float:
            print(text)
            continue
        else:
            yield tokenizer.tokenize(text)
