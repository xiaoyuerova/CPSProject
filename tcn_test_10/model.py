import copy

import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_10.data_tcn.parameters import Parameters
from tcn_test_10.utils import generate_mask, transform_mask

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)


class TextTransform(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_size):
        super(TextTransform, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim * parameters.Sentence_max_length, output_size)
        self.fc = nn.Sequential(
            self.linear,
            # nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1),
            # nn.ReLU(),
        )
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def forward(self, x):
        """

        :param offsets:
        :param x: torch.tenser() [batch_size, sequence_length * window]
        :return: torch.Tensor [bz, window, embed_size]
        """
        output = self.embedding(x)
        output = self.fc(output.view([len(x), -1]))
        return output.argmax(1)


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
        self.textTransform = TextTransform(vocab_size, parameters.word_embedding_size, parameters.sentence_output_size)
        sentence_size = parameters.sentence_output_size + 8
        self.embedding = nn.Embedding(sentence_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)
        # self.drop = nn.Dropout(emb_dropout)
        # self.emb_dropout = emb_dropout
        self.init_weights()
        # print('vocab_size', vocab_size)

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.normal_(0, 0.01)
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()

    def forward(self, x):
        """"""
        text_loader, mask = generate_mask(x)
        text_actions = []
        for text in text_loader:
            text_actions.extend(self.textTransform(text))
        # print('mask', len(mask), mask)
        # print('text_actions', len(text_actions))
        # print(x)
        x, offsets = transform_mask(x, text_actions, copy.deepcopy(mask))

        emb = self.embedding(x)
        # print('emb', emb.size())
        y = []
        for i in range(len(offsets) - 1):
            sequence = emb[offsets[i]: offsets[i + 1]]
            sequence = sequence.view([1, -1, parameters.sentence_embedding_size])
            temp = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
            temp = self.decoder(temp)
            y.append(temp[0])
        # print('y[1]', y[1].size())

        yy = []
        # print(mask)
        for i in range(len(y)):
            t = []
            for j in range(len(y[i])):
                if mask.pop(0):
                    t.append(y[i][j])
            t = [item.view([1, -1]) for item in t]
            if len(t) > 0:
                t = torch.cat(t)
                # print(t.size())
                yy.append(t)
        return yy
