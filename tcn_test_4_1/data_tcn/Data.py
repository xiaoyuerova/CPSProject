import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator
from tcn_test_4_1.data_tcn.parameters import Parameters
from tcn_test_4_1.data_tcn.MyDataset import MyDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
parameters = Parameters()


def text_tokenizer(sentence):
    if type(sentence) != str:
        print('type error', sentence)
    tokens = tokenizer.tokenize(sentence)
    length = len(tokens)
    if length < parameters.Sentence_max_length - 2:
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens = tokens + ['[PAD]'] * (parameters.Sentence_max_length - length - 2)
    else:
        tokens = tokens[:parameters.Sentence_max_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
    return tokens


def label_pipeline(label):
    return int(label)


def yield_tokens(d_iter):
    for _, text in d_iter:
        if type(text) == float:
            print(text)
            continue
        else:
            yield tokenizer.tokenize(text)


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['[PAD]', '[CLS]', '[SEP]'])
    vocab.set_default_index(vocab["[PAD]"])
    return vocab


def data_iter(dataset: MyDataset):
    for index in range(0, dataset.__len__()):
        yield dataset.__getitem__(index)


class Data:
    def __init__(self, dataset):
        """

        :param dataset:
        """
        self.vocab = build_vocab_from_iterator_re(data_iter(dataset))
        self.text_pipeline = lambda x: self.vocab(text_tokenizer(x))
        self.label_pipeline = label_pipeline
        self.device = parameters.device
        self.dataloader = DataLoader(dataset, batch_size=parameters.Batch_size, shuffle=False,
                                     collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            text_list.append(self.text_pipeline(_text))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        return label_list.to(self.device), text_list.to(self.device)
