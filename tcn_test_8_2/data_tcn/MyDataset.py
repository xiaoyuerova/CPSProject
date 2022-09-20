import torch.utils.data as data
import pandas as pd
from tcn_test_8_2.data_tcn.parameters import Parameters

parameters = Parameters()


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, df: pd.DataFrame):
        """

        :param df:
        read_index: 存储一组数据中心词的index
        padding: [int,int] 中心词前后不够取时，作padding. padding[0]表示左边需要padding的数量，padding[1]表示右边
        """
        self.df = df
        self.grouped_data = df.groupby(['NewName', 'LevelNumber'])
        self.names = []
        self.select = []  # 选择chat数据
        self.length = 0
        self.init()

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据

        :returns labels, sequence ：一个滑动窗口的标签和数据(做padding)
        """
        group = self.grouped_data.get_group(self.names[index])
        return group['Label'], group['Script'], self.select[index]

    def __len__(self):
        """
        返回数据集的长度

        :return 滑动窗口的个数
        """
        return self.length

    def init(self):
        for name, group in self.grouped_data:
            self.names.append(name)
            _list = []
            for idx, index in enumerate(group.index):
                if group.at[index, 'DataCode'] == 5000:
                    _list.append(idx)
            self.select.append(_list)
        self.length = len(self.names)
