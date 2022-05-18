import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_5.data_tcn.parameters import Parameters
from tcn_test_5.utils import generate_mask, divide_data_for_tcn

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()


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
        return tokens


class CpsTcnModel(nn.Module):
    def __init__(self, input_size, output_size, levels: [int],
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
        self.encoder = Embedding()
        # 固定预训练模型参数
        for p in self.parameters():
            p.requires_grad = False
        self.tcn_for_center = TemporalConvNet(input_size, [input_size] * levels[0], kernel_size, dropout=dropout)
        self.tcn_for_outer = TemporalConvNet(input_size, [input_size] * levels[1], kernel_size, dropout=dropout)

        self.decoder = nn.Linear(input_size*3, output_size)

        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, _input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb_tokens = self.drop(self.encoder(_input))
        # print('emb', emb.size())
        tokens_before, tokens_center, tokens_behind = divide_data_for_tcn(emb_tokens)
        y_before = self.tcn_for_outer(tokens_before.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y_center = self.tcn_for_center(tokens_center.transpose(1, 2)).transpose(1, 2)
        y_behind = self.tcn_for_outer(tokens_behind.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y = torch.cat([y_before, y_center, y_behind], dim=1)
        # print('ytcn', y.size())
        y = self.decoder(y)
        # print('yliner', y.size())
        return y
