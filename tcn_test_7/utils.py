# from pytorch_pretrained_bert import BertTokenizer
import copy

from transformers import BertTokenizer
from tcn_test_7.data_tcn import *
from torchtext.vocab import build_vocab_from_iterator, Vocab
import pandas as pd
from sklearn.model_selection import KFold
from tcn_test_7.model import CpsTcnModel

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


def load_data():
    _dir = '../data/tcn_test_data/tcn-model-data3.csv'
    df = pd.read_csv(_dir)
    df = df[df['DataCode'] == 5000]
    df = df[df['Action_S'].notna()]

    # 按组划分测试数据
    grouped_data = df.groupby(['NewName'])
    df_list = []
    for idx, (name, group) in enumerate(grouped_data):
        df_list.append(group)
    return df_list


def get_vocab():
    _dir = '../data/tcn_test_data/tcn-model-data3.csv'
    df = pd.read_csv(_dir)
    df = df[df['DataCode'] == 5000]
    df = df[df['Action_S'].notna()]
    df.reset_index(inplace=True)
    dataset = MyDataset(df)
    return build_vocab_from_iterator_re(data_iter(dataset))


# k折交叉验证
def k_fold_test(vocab: Vocab):
    # 加载数据
    df_list = load_data()
    kf = KFold(n_splits=parameters.n_splits, shuffle=True, random_state=parameters.random_state)
    for train_index, test_index in kf.split(df_list):
        # 数据是df
        train_data = pd.concat([df_list[i] for i in train_index])
        test_data = pd.concat([df_list[i] for i in test_index])
        train_data.reset_index(inplace=True)
        test_data.reset_index(inplace=True)

        # 数据是MyDataset
        dataset_train = MyDataset(train_data)
        dataset_test = MyDataset(test_data)

        # 数据是Data
        train_data = Data(dataset_train, vocab)
        test_data = Data(dataset_test, vocab)
        yield train_data, test_data


def generate_model(vocab):
    model = CpsTcnModel(len(vocab), 11, [parameters.embedding_size] * 3)
    model.to(parameters.device)
    return model


if __name__ == '__main__':
    pass
