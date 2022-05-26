import math

import torch
import torch.nn as nn
from source.TCN.tcn import TemporalConvNet
from tcn_test_6_2.data_tcn.parameters import Parameters

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
            output = self.embeddingBag(x[i], offsets[i])[parameters.Sliding_window_radius]
            outputs.append(output)
        return torch.cat([item.view(1, parameters.embedding_size) for item in outputs], dim=0)


class CpsTcnModel2(nn.Module):
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
        super(CpsTcnModel2, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        # self.tcn = TemporalConvNet(embed_dim, [embed_dim] * levels, kernel_size, dropout=dropout)

        self.linear = nn.Linear(embed_dim, output_size)
        self.fc = nn.Sequential(
            self.linear,
            nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
        )
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def forward(self, texts, offsets):
        emb_tokens = self.embedding(texts, offsets)

        y = self.fc(emb_tokens)

        return y
