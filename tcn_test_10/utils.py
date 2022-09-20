# from pytorch_pretrained_bert import BertTokenizer
import math
import os

import torch
from transformers import BertTokenizer
from tcn_test_10.data_tcn.MyDataset import MyDataset
from tcn_test_10.data_tcn.parameters import Parameters
from torchtext.vocab import build_vocab_from_iterator


path = os.path.join(os.path.dirname(__file__), './model/bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(path)
parameters = Parameters()
torch.autograd.set_detect_anomaly(True)


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['[PAD]', '[CLS]', '[SEP]'])
    vocab.set_default_index(vocab["[PAD]"])
    return vocab


def data_iter(dataset: MyDataset):
    for index in range(0, dataset.__len__()):
        _, actions = dataset.__getitem__(index)
        for text in actions:
            if type(text) == str:
                yield text


def yield_tokens(d_iter):
    for text in d_iter:
        if type(text) == float:
            print(text)
            continue
        else:
            yield tokenizer.tokenize(text)


def generate_mask(x: list):
    mask = []
    text = []
    for batch in x:
        for item in batch:
            if type(item) == torch.Tensor:
                text.append(item)
                mask.append(True)
            else:
                mask.append(False)
    return text_loader(text), mask


def text_loader(text: list):
    size = 16
    length = math.ceil(len(text) / size)
    for i in range(length):
        data = text[i * size: (i + 1) * size]
        yield torch.cat(data).view([-1, parameters.Sentence_max_length]).contiguous()


def transform_mask(x, text_actions: list, mask: list):
    # print('transform_mask mask', mask)
    for i in range(len(x)):
        # print(len(x[i]))
        for j in range(len(x[i])):
            if mask.pop(0):
                x[i][j] = text_actions.pop(0)
        # print('x[i]', x[i])
        x[i] = torch.tensor(x[i], dtype=torch.int64).to(parameters.device)
    # padding
    offsets = [0]
    for i in range(len(x)):
        offsets.append(len(x[i]) + offsets[-1])
    x = torch.cat(x)
    offsets = torch.tensor(offsets, dtype=torch.int64).to(parameters.device)
    return x, offsets
