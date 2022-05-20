# from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from tcn_test_0.data_tcn.MyDataset import MyDataset
from tcn_test_0.data_tcn.parameters import Parameters
from torchtext.vocab import build_vocab_from_iterator


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
# tokenizer = get_tokenizer('basic_english')
parameters = Parameters()


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
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
            yield tokenizer(text)
