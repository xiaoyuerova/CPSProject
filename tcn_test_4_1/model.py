import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_4_1.data_tcn.parameters import Parameters

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
        mask = generate_mask(x).to(parameters.device)
        tokens = self.bert_model(x, attention_mask=mask)[0]
        return tokens


class CpsTcnModel(nn.Module):
    def __init__(self, vocab_size, output_size, num_channels, input_size=parameters.embedding_size,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
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
        self.encoder = nn.Embedding(vocab_size, input_size)

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
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)
        # init_range = 0.5
        # self.encoder.weight.data.uniform_(-init_range, init_range)
        # self.decoder.weight.data.uniform_(-init_range, init_range)
        # self.decoder.bias.data.zero_()

    def forward(self, _input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(_input))
        # print('emb', emb.size())
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = y.mean(dim=1)
        # print('ytcn', y.size())
        y = self.decoder(y)
        # print('yliner', y.size())
        return y
