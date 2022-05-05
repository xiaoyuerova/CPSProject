import numpy as np
import pandas as pd
import torch

from source.data_tcn import *


def main():
    _dir = './data/tcn-model-data-mini.csv'
    df = pd.read_csv(_dir)
    df = df[df['Action'].notna()]
    df.reset_index(inplace=True)
    print(df[540:542])
    dataset = MyDataset(df)
    data = Data(dataset)
    for batch in data.dataloader:
        i = 0
        if i < 1:
            for c in batch:
                print(c.size())
            i += 1


if __name__ == '__main__':
    main()
