import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from source.data_tcn_2.parameters import Parameters

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()


def generate_mask(x: torch.Tensor):
    mask = []
    for sentence in x:
        temp = [1 if t != 0 else 0 for t in sentence]
        mask.append(temp)
    return torch.tensor(mask, dtype=torch.int64)


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x: torch.Tensor):
        """

        :param x: torch.tenser() [batch_size, sequence_length,sentence_length]
        :return:
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, -1)
        mask = generate_mask(x).to(parameters.device)
        tokens = self.bert_model(x, attention_mask=mask)[0]
        tokens = tokens.view(batch_size, seq_length, -1, 768)
        # tokens = torch.mean(tokens, dim=2)
        temp = []
        for batch in range(batch_size):
            t = tokens[batch]
            temp.append(torch.cat([t[index] for index in range(t.size(0))], dim=0).view(1, -1, 768))
        tokens = torch.cat(temp, dim=0)
        # print(tokens.size())
        return tokens


class CpsTcnModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        """
        这个模型是一开始想错了，将一句话压缩成一个单词：[?,  n,  sentence_n,  k]. mean(dim= 2 )  ->  [?,  n,  k]
        :param input_size: embedding后的维度
        :param output_size: 可能的输出类别个数（11）
        :param num_channels:
        :param kernel_size:
        :param dropout:
        :param emb_dropout:
        :param tied_weights:
        """
        super(CpsTcnModel, self).__init__()
        self.encoder = Embedding()
        # 固定预训练模型参数
        for p in self.parameters():
            p.requires_grad = False
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, _input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(_input))
        # print('emb', emb.size())
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = y.contiguous()[:, -1]
        # print('ytcn', y.size())
        y = self.decoder(y)
        # print('yliner', y.size())
        return y
