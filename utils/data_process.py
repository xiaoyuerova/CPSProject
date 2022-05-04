import pandas as pd
import torch

from source.data import *
from utils.TextClassification import TextClassification


# train_iter = AG_NEWS(split='train')


def data_process():
    data_dir = '../data/trainSetMini2.csv'
    df = pd.read_csv(data_dir)
    df2 = df[['Label2', 'Text']]
    dataset = MyDataset(df2)
    data = Data(dataset)
    num_class = len(set([label for label in df['Label']]))
    vocab_size = len(data.vocab)
    model = TextClassification(vocab_size, num_class).to(data.device)



def test():
    data_process()


if __name__ == '__main__':
    test()
