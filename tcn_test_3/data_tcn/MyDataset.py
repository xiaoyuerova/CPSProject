import torch.utils.data as data
import pandas as pd
from tcn_test_3.data_tcn.parameters import Parameters

parameters = Parameters()


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, df: pd.DataFrame):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        self.df = df
        self.length = 0
        self.read_index = []
        self.window = parameters.Sliding_window_radius * 2
        self.init()

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）

        :returns labels, sequence ：一个滑动窗口的标签和数据
        """
        sequence = []
        for i in range(self.window):
            sequence.append(self.df.loc[self.read_index[index] + i, 'Label'])
        label = self.df['Label'][self.read_index[index] + self.window]
        return label, sequence

    def __len__(self):
        """
        返回数据集的长度

        :return 滑动窗口的个数
        """
        return self.length

    def init(self):
        grouped_data = self.df.groupby(['NewName', 'LevelNumber'])
        for name, group in grouped_data:
            for index in group.index:
                if group['DataCode'][index] == 5000:
                    df_cell = group.loc[(index - self.window): index-1]
                    if len(df_cell) < self.window:
                        continue
                    self.length += 1
                    self.read_index.append(index - self.window)
