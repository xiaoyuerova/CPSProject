import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_5_1.data_tcn.parameters import Parameters
from tcn_test_5_1.utils import generate_mask, divide_data_for_tcn, to_tenser

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens(parameters.special_tokens)


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.bert_model: BertModel = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.resize_token_embeddings(len(tokenizer))

    def forward(self, x: torch.Tensor):
        """

        :param x: torch.tenser() [batch_size, sequence_length,sentence_length]
        :return:
        """
        mask = generate_mask(x).to(parameters.device)
        tokens = self.bert_model(x, attention_mask=mask)[0]
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

        self.linear_for_center = nn.Linear(input_size, output_size)
        self.linear_for_outer = nn.Linear(input_size, output_size)

        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.linear_for_center.bias.data.fill_(0)
        self.linear_for_center.weight.data.normal_(0, 0.01)
        self.linear_for_outer.bias.data.fill_(0)
        self.linear_for_outer.weight.data.normal_(0, 0.01)

    def forward(self, _input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # print(to_tenser(_input, 0).size())
        tokens_before = self.encoder(to_tenser(_input, 0))
        tokens_center = self.encoder(to_tenser(_input, 1))
        tokens_behind = self.encoder(to_tenser(_input, 2))
        y_before = self.tcn_for_outer(tokens_before.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y_center = self.tcn_for_center(tokens_center.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y_behind = self.tcn_for_outer(tokens_behind.transpose(1, 2)).transpose(1, 2).mean(dim=1)
        y_before = self.linear_for_outer(y_before)
        y_center = self.linear_for_center(y_center)
        y_behind = self.linear_for_outer(y_behind)

        y = (y_center + y_before * y_before + y_center * y_behind) / 3
        return y
