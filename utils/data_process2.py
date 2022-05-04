# tokenizer = lambda x: x.split() 如果语料已经全部是处理好的句子，直接分词就可以了
# {'SSI-O': 0, 'SESU-P': 1, 'SESU-A': 2, 'CP-S': 3, 'CMC-S': 4, 'SN-A': 5, 'SMC-O': 6, 'CEC-R': 7, 'SMC-R': 8, 'CEC-S': 9, 'SSI-U': 10, 'SN-D': 11, 'CMC-G': 12, 'CRF-F': 13, 'CP-G': 14, 'CRF-R': 15, 'SMC-I': 16, 'SSI-T': 17, 'CE-E': 18, 'CM-G': 19}

import spacy
from torchtext import data
import torch
from torch import nn
from torch import optim
from simple_LSTM import SimpleLSTM, train_val_test

import pandas as pd


def test():
    data_dir = '../data/trainSetMini.csv'
    data_set = pd.read_csv(data_dir)
    label_dict = {'SSI-O': 0, 'SESU-P': 1, 'SESU-A': 2, 'CP-S': 3, 'CMC-S': 4, 'SN-A': 5, 'SMC-O': 6, 'CEC-R': 7,
                  'SMC-R': 8, 'CEC-S': 9, 'SSI-U': 10, 'SN-D': 11, 'CMC-G': 12, 'CRF-F': 13, 'CP-G': 14, 'CRF-R': 15,
                  'SMC-I': 16, 'SSI-T': 17, 'CE-E': 18, 'CM-G': 19}
    data_set = data_set[['Text', 'Label']]
    data_set['Label2'] = data_set['Label'].astype('category')
    data_set['Label2'] = data_set['Label2'].cat.codes
    data_set.to_csv('../data/trainSetMini2.csv')

    print(data_set.head())


def tokenizer(text):
    spacy_en = spacy.load('en_core_web_sm')
    return [toke.text for toke in spacy_en.tokenizer(text)]


# 也可以直接在Field里用 tokenize='spacy'，效果等同于上面自定义的tokenizer函数，只不过需要先link好，这里不展开说了
# REVIEW 用来存储用户评论，include_lengths 设为 True 方便后续使用 pack_padded_sequence


def data_process():
    data_dir = '../data/trainSetMini2.csv'
    REVIEW = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True)
    POLARITY = data.LabelField(sequential=False, dtype=torch.float)
    # fields字典里，元组的第一个元素将成为接下来从数据sample出的每个batch的属性，里面存放着对应的数据，第二个元素是对应的Field
    fields = [('null', None), ('dialogue', REVIEW), ('label0', None), ('label', POLARITY)]
    dialogue_dataset = data.TabularDataset(
        path=data_dir,
        format='CSV',
        fields=fields,
        skip_header=True
    )
    print('dialogue_dataset ok')
    vars(dialogue_dataset.examples[10])
    train_set, test_set, val_set = dialogue_dataset.split(split_ratio=[0.8, 0.1, 0.1], stratified=True,
                                                          strata_field='label')
    vocab_size = 1000
    REVIEW.build_vocab(train_set, max_size=vocab_size)
    POLARITY.build_vocab(train_set)

    device = 'cpu'
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train_set, val_set, test_set),
                                                                 batch_size=32,
                                                                 device=device,
                                                                 sort_within_batch=True,
                                                                 sort_key=lambda x: len(x.dialogue))

    lstm_model = SimpleLSTM(hidden_size=256, embedding_dim=100, vocab_size=1002)
    lstm_model.to(device)

    # 优化器
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    # 损失函数
    criterion = nn.CrossEntropyLoss()  # 多分类 （负面、正面、中性）

    train_val_test(lstm_model, optimizer, criterion, train_iter, val_iter, test_iter, 5)


if __name__ == '__main__':
    test()
