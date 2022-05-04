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
    data_dir = 'data/fenbu/filtered_data2.csv'
    df = pd.read_csv(data_dir)
    df2 = df[['Label', 'Chat']]
    df2 = df2[df2['Label'] != 0.0]
    df2.reset_index(drop=True, inplace=True)
    dataset = MyDataset(df2)
    data = Data(dataset)

    num_class = len(set([label for label in df['Label']]))
    vocab_size = len(data.vocab)
    model = TextClassification(vocab_size, num_class).to(data.device)
    train_model(model, data.dataloader, data.dataloader, data.dataloader)

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


if __name__ == '__main__':
    main()
