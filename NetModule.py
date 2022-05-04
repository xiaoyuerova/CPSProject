import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sort_batch(data, label, length):
    batch_size = data.size(0)
    # 先将数据转化为numpy()，再得到排序的index
    inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data = data[inx]
    label = label[inx]
    length = length[inx]
    # length转化为了list格式，不再使用torch.Tensor格式
    length = list(length.numpy())
    length = torch.FloatTensor(length)
    return data, label, length


class NetModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bi_flag, dropout=0.5):
        # input_dim 输入特征维度 d_input
        # hidden_dim 隐藏层的大小
        # output_dim 输出层的大小（分类的类别数）
        # num_layers LSTM 隐藏层的层数
        # biFlag 是否使用双向
        super(NetModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if bi_flag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = bi_flag
        # 根据需要修改device
        self.device = torch.device("cuda")

        # 定义LSTM网络的输入，输出，层数，是否batch_first，dropout比例，是否双向
        self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout, bidirectional=bi_flag)
        # 定义线性分类层，使用logsoftmax输出
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim * self.bi_num, output_dim),
            nn.LogSoftmax(dim=2)
        )

        self.to(self.device)

    def init_hidden(self, batch_size):
        # 定义初始的hidden state
        return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device))

    def forward(self, x, y, length):
        # 输入原始数据x，标签y，以及长度length
        # 准备
        batch_size = x.size(0)
        max_length = torch.max(length)
        # 根据最大长度截断
        x = x[:, 0:max_length, :]
        y = y[:, 0:max_length]
        x, y, length = sort_batch(x, y, length)
        x, y = x.to(self.device), y.to(self.device)
        # pack sequence
        x = pack_padded_sequence(x, length, batch_first=True)

        # run the network
        hidden1 = self.init_hidden(batch_size)
        out, hidden1 = self.layer1(x, hidden1)
        # out,_=self.layerLSTM(x) is also ok if you don't want to refer to hidden state
        # unpack sequence
        out, length = pad_packed_sequence(out, batch_first=True)
        out = self.layer2(out)
        # 返回正确的标签，预测标签，以及长度向量
        return y, out, length
