import torch.utils.data as data


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, inputs, masks, labels):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）

        :returns labels, sequence ：一个滑动窗口的标签和数据
        """
        return {'input_ids': self.inputs[index],
                'attention_mask': self.masks[index],
                'labels': self.labels[index]
                }
        # return self.inputs[index], self.masks[index], self.labels[index]

    def __len__(self):
        """
        返回数据集的长度

        :return 滑动窗口的个数
        """
        return len(self.labels)
