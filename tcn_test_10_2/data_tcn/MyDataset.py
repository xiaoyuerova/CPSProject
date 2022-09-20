import torch.utils.data as data
import pandas as pd
from tcn_test_10_2.data_tcn.parameters import Parameters

parameters = Parameters()


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, df: pd.DataFrame):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        self.grouped_data = df.groupby(['NewName', 'LevelNumber'])
        self.names = []
        self.length = 0
        self.init()

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）

        :returns labels, sequence ：一个滑动窗口的标签和数据
        """
        group = self.grouped_data.get_group(self.names[index])
        actions = group['Action'].to_list()
        labels = group['Label'].to_list()
        # print('my dataset', len(labels))
        return labels, actions

    def __len__(self):
        """
        返回数据集的长度

        :return 滑动窗口的个数
        """
        return self.length

    def init(self):
        for name, group in self.grouped_data:
            if len(group[group.DataCode == 5000]) > 0:
                self.length += 1
                self.names.append(name)
