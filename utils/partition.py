import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def data_split(data: pd.DataFrame, labels: pd.DataFrame, test_prop, valid_prop=None):
    """
    划分数据集
    :param data:
    :param labels:
    :param test_prop:
    :param valid_prop:
    :return:
    """

    data = pd.read_csv('../data.csv')
    x = data['Texts'].astype('str')
    y = data['Labels']
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
