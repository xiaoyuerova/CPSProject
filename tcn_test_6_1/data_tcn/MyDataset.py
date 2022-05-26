import torch.utils.data as data
import pandas as pd
from tcn_test_6_1.data_tcn.parameters import Parameters

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
        self.length = 0
        self.read_index = []
        self.padding = []
        self.sliding_window_radius = parameters.Sliding_window_radius
        self.window = parameters.Sliding_window_radius * 2 + 1
        self.init()

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据

        :returns labels, sequence ：一个滑动窗口的标签和数据(做padding)
        """
        read_index = self.read_index[index]
        sequence = []
        if self.padding[index] is None:
            # 无需padding
            for i in range(self.window):
                sequence.append(self.df.at[read_index - self.sliding_window_radius + i, 'Action_S'])
        else:
            if self.padding[index][0] is None:
                # 在序列后面padding
                for i in range(self.window):
                    if i < self.window - self.padding[index][1]:
                        sequence.append(self.df.at[read_index - self.sliding_window_radius + i, 'Action_S'])
                    else:
                        sequence.append('[PAD]')
            else:
                # 在序列前面padding
                for i in range(self.window):
                    if i >= self.padding[index][0]:
                        sequence.append(self.df.at[read_index - self.sliding_window_radius + i, 'Action_S'])
                    else:
                        sequence.append('[PAD]')
        label = self.df['Label'][self.read_index[index]]
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
            group_index_start = group.index[0] + self.sliding_window_radius
            group_index_end = group.index[-1] - self.sliding_window_radius
            for index in group.index:
                if group['DataCode'][index] == 5000:
                    self.length += 1
                    self.read_index.append(index)

                    # 计算padding
                    if group_index_start <= index <= group_index_end:
                        self.padding.append(None)
                    elif index < group_index_start and index <= group_index_end:
                        padding_size = group_index_start - index
                        self.padding.append([padding_size, None])
                    elif index > group_index_end and index >= group_index_start:
                        padding_size = index - group_index_end
                        self.padding.append([None, padding_size])
                    else:
                        self.length -= 1
                        self.read_index.pop()
                        print('MyDataset init wrong!')
                        print('index:', index)
                        print('group_index_start:', group_index_start)
                        print('group_index_end:', group_index_end)
