import numpy as np
import pandas as pd
import torch

from source.data import *
from source.model import *


def main():
    # data_dir = './data/trainSetMini2.csv'
    # df = pd.read_csv(data_dir)
    # df2 = df[['Label2', 'Text']]
    # dataset = MyDataset(df2)
    # data = Data(dataset)
    #
    # num_class = len(set([label for label in df['Label']]))
    # print(num_class)
    # vocab_size = len(data.vocab)
    # model = TextClassification(vocab_size, num_class).to(data.device)
    #
    # train_model(model, data.dataloader, data.dataloader, data.dataloader)

    # 实验一
    # data_dir = 'data/fenbu/filtered_data.csv'
    # df = pd.read_csv(data_dir)
    # df2 = df[['Label', 'DataDesc']]
    # dataset = MyDataset(df2)
    # data = Data(dataset)
    #
    # num_class = len(set([label for label in df['Label']]))
    # vocab_size = len(data.vocab)
    # model = TextClassification(vocab_size, num_class).to(data.device)
    # train_model(model, data.dataloader, data.dataloader, data.dataloader)

    # 实验二
    # data_dir = 'data/fenbu/filtered_data2.csv'
    # df = pd.read_csv(data_dir)
    # df2 = df[['Label', 'Chat']]
    # df2 = df2[df2['Label'] != 0.0]
    # df2.reset_index(drop=True, inplace=True)
    # dataset = MyDataset(df2)
    # data = Data(dataset)
    #
    # num_class = len(set([label for label in df['Label']]))
    # vocab_size = len(data.vocab)
    # model = TextClassification(vocab_size, num_class).to(data.device)
    # train_model(model, data.dataloader, data.dataloader, data.dataloader)

    # 实验三
    # data_dir = 'data/fenbu/filtered_data3.csv'
    # df = pd.read_csv(data_dir)
    # df2 = df[['Label', 'Chat']]
    # df2 = df2[df2['Label'] != 0.0]
    # df2.reset_index(drop=True, inplace=True)
    # dataset = MyDataset(df2)
    # data = Data(dataset)
    #
    # num_class = len(set([label for label in df['Label']]))
    # vocab_size = len(data.vocab)
    # model = TextClassification(vocab_size, num_class).to(data.device)
    # train_model(model, data.dataloader, data.dataloader, data.dataloader)

    _dir = './data/tcn_test_data/tcn-model-data3.csv'
    df = pd.read_csv(_dir)
    df = df[df['DataCode'] == 5000]
    df = df[df['Action_S'].notna()]

    # 按组划分测试数据
    grouped_data = df.groupby(['NewName'])
    divide = int(len(grouped_data) * 0.8)
    df_list1 = []
    df_list2 = []
    for idx, (name, group) in enumerate(grouped_data):
        if idx < divide:
            df_list1.append(group)
        else:
            df_list2.append(group)
    df_train = pd.concat(df_list1)
    df_test = pd.concat(df_list2)

    # 初始化数据
    df_train.reset_index(inplace=True)
    dataset_train = MyDataset(df_train)
    data_train = Data(dataset_train)
    vocab_size = len(data_train.vocab)

    df_test.reset_index(inplace=True)
    dataset_test = MyDataset(df_test)
    data_test = Data(dataset_test)

    num_class = 11
    model = TextClassification(vocab_size, num_class).to(data_train.device)
    total = 0
    total2 = 0
    for param in model.parameters():
        total += param.nelement()
        if param.requires_grad:
            total2 += param.nelement()
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("Number of training parameter: %.2fM" % (total2 / 1e6))
    train_model(model, data_train.dataloader, data_test.dataloader, data_test.dataloader)


if __name__ == '__main__':
    main()
