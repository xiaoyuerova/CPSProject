# from pytorch_pretrained_bert import BertTokenizer
import math
import os

import pandas as pd
import torch
from transformers import BertTokenizer
from tcn_test_10_1.data_tcn.MyDataset import MyDataset
from tcn_test_10_1.data_tcn.parameters import Parameters
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


def pre_process(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model = torch.load('../tcn_test_4_1_3/model/model.pkl')
    vocab = torch.load('../tcn_test_4_1_3/model/vocab.pkl')

    df = df_train[df_train.DataCode == 5000]

    bz = 0
    indexes = []
    texts = []
    for index in df.index:
        text = df_train.at[index, 'Action']
        text = text_tokenizer(text)
        text = vocab(text)
        bz += 1
        indexes.append(index)
        texts.append(text)
        if bz == 64 or index == df.index[-1]:
            texts = torch.tensor(texts, dtype=torch.int64).to(parameters.device)
            output = model(texts)
            predict = output.argmax(1)
            predict = [9 if p == 6 else (10 if p == 7 else p) for p in predict]
            for i, p in enumerate(predict):
                df_train.at[indexes[i], 'Action'] = p
            bz = 0
            indexes = []
            texts = []

    df = df_test[df_test.DataCode == 5000]
    for index in df.index:
        text = df_test.at[index, 'Action']
        text = text_tokenizer(text)
        text = vocab(text)
        bz += 1
        indexes.append(index)
        texts.append(text)
        if bz == 64 or index == df.index[-1]:
            texts = torch.tensor(texts, dtype=torch.int64).to(parameters.device)
            output = model(texts)
            predict = output.argmax(1)
            predict = [9 if p == 6 else (10 if p == 7 else p) for p in predict]
            for i, p in enumerate(predict):
                df_test.at[indexes[i], 'Action'] = p
            bz = 0
            indexes = []
            texts = []

    return df_train, df_test
