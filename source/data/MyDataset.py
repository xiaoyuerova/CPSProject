import torch.utils.data as data
import pandas as pd


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, df: pd.DataFrame):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        self.df = df
        pass

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）
        2. 对读取到的数据进行数据增强 (数据增强是深度学习中经常用到的，可以提高模型的泛化能力)
        3. 返回数据对 （一般我们要返回 图片，对应的标签） 在这里因为我没有写完整的代码，返回值用 0 代替
        """
        return (self.df[c][index] for c in self.df)

    def __len__(self):
        """
        返回数据集的长度
        """
        return self.df.__len__()
