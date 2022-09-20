import copy

import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_10_1.data_tcn.parameters import Parameters
from tcn_test_10_1.utils import generate_mask, transform_mask

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)


class TextTransform(nn.Module):
    def __init__(self, vocab_size, num_class, embed_dim=parameters.word_embedding_size):
        super(TextTransform, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        text torch.Size([44])
        embedded torch.Size([8, 100])
        output torch.Size([8, 12])
        embeddingBag 会对每个bag进行sum操作
        """
        embedded = self.embedding(text, offsets)
        output = self.fc(embedded)
        return output


class CpsTcnModel(nn.Module):
    def __init__(self, vocab_size, output_size, num_channels, input_size=parameters.sentence_embedding_size,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        """
        :param input_size: embedding后的维度
        :param output_size: 可能的输出类别个数（11）
        :param num_channels:
        :param kernel_size:
        :param dropout:
        :param emb_dropout:
        :param tied_weights:
        """
        super(CpsTcnModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()

    def forward(self, x):
        """"""
        emb = self.embedding(x)
        # print('emb', emb.size())
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        # print('y', y.size())
        y = self.decoder(y)
        # print('y', y.size())
        return y[0]
