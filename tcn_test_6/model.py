import math

import torch
import torch.nn as nn
from source.TCN.tcn import TemporalConvNet
from tcn_test_6.data_tcn.parameters import Parameters
from tcn_test_6.utils import generate_mask, divide_data_for_tcn

parameters = Parameters()


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, sparse=True):
        super(Embedding, self).__init__()
        self.embeddingBag = nn.EmbeddingBag(vocab_size, embed_dim, sparse=sparse)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embeddingBag.weight.data.uniform_(-init_range, init_range)

    def forward(self, x: [torch.Tensor], offsets):
        """

        :param offsets:
        :param x: torch.tenser() [batch_size, sequence_length * window]
        :return: torch.Tensor [bz, window, embed_size]
        """
        batch_size = len(x)
        outputs = []
        for i in range(batch_size):
            output = self.embeddingBag(x[i], offsets[i])
            # output: torch.tensor [window, embed_size]
            outputs.append(output)
        return torch.cat([item.view(1, -1, parameters.embedding_size) for item in outputs], dim=0)


class CpsTcnModel(nn.Module):
    def __init__(self, vocab_size, output_size, levels, embed_dim=parameters.embedding_size,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        """

        :param input_size: embedding后的维度
        :param output_size: 可能的输出类别个数（11）
        :param levels: tcn中 block的层数
        :param kernel_size:
        :param dropout:
        :param emb_dropout:
        """
        super(CpsTcnModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.tcn = TemporalConvNet(embed_dim, [embed_dim] * levels, kernel_size, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, output_size)
        self.linear2 = nn.Linear(embed_dim, output_size)
        self.linear_for_outer = nn.Sequential(
            self.linear1,
            nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
        )
        self.linear_for_center = nn.Sequential(
            self.linear2,
            nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
        )

        self.exp_before = 2
        self.exp_behind = 2
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.linear1.weight.data.uniform_(-init_range, init_range)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.uniform_(-init_range, init_range)
        self.linear2.bias.data.zero_()

    def forward(self, texts, offsets):
        emb_tokens = self.embedding(texts, offsets)
        # print('emb_tokens', emb_tokens.size())
        tokens_before, tokens_center, tokens_behind = divide_data_for_tcn(emb_tokens)
        # print('tokens_before', tokens_before.size())
        y_before = self.tcn(tokens_before.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y_behind = self.tcn(tokens_behind.transpose(1, 2)).transpose(1, 2).mean(dim=1)

        y_before = self.linear_for_outer(y_before)
        y_center = self.linear_for_center(tokens_center)
        y_behind = self.linear_for_outer(y_behind)

        return (y_before.pow(self.exp_before) + y_center + y_behind.pow(self.exp_behind))/3
