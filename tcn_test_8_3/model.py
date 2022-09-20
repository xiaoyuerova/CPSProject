import math

import torch
import torch.nn as nn
from source.TCN.tcn import TemporalConvNet
from tcn_test_8_3.data_tcn.parameters import Parameters

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
            outputs.append(output)
        # 其实这里bz=1，因为要作为tensor处理，序列长度不一致
        return torch.cat([item.view(1, -1, parameters.embedding_size) for item in outputs], dim=0)


class TcnForSingle(nn.Module):
    def __init__(self, vocab_size, levels, embed_dim, kernel_size=2, dropout=0.3):
        super(TcnForSingle, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.tcn = TemporalConvNet(embed_dim, [embed_dim] * levels, kernel_size, dropout=dropout)
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)

    def forward(self, texts, offsets):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.encoder(texts)
        y = []
        offsets = offsets[0]
        for i in range(len(offsets)-1):
            sentence = emb[:, offsets[i]:offsets[i+1]]
            y.append(self.tcn(sentence.transpose(1, 2)).transpose(1, 2).mean(dim=1))
        sentence = emb[:, offsets[-1]:]
        y.append(self.tcn(sentence.transpose(1, 2)).transpose(1, 2).mean(dim=1))
        y = torch.cat(y, dim=0)
        return y


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
        # self.tcn_for_single = TcnForSingle(vocab_size, levels, embed_dim)
        self.tcn = TemporalConvNet(embed_dim, [embed_dim] * levels, kernel_size, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_size),
            nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1),
            nn.ReLU(),
        )

    def forward(self, texts, offsets):
        emb_tokens = self.embedding(texts, offsets)
        y_sequence = self.tcn(emb_tokens.transpose(1, 2)).transpose(1, 2)[0]
        # y_single = self.tcn_for_single(texts, offsets)
        y = self.fc(0.0*y_sequence + 1.0*emb_tokens[0])
        return y
