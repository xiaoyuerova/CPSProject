import copy

import torch
import torch.nn as nn
from transformers import logging
from transformers import BertModel, BertTokenizer
from source.TCN.tcn import TemporalConvNet
from tcn_test_10_3.data_tcn.parameters import Parameters
from tcn_test_10_3.utils import generate_mask, transform_mask
from torchcrf import CRF

parameters = Parameters()
logging.set_verbosity_warning()
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)


class CpsTcnModel(nn.Module):
    def __init__(self, vocab_size, output_size, num_channels, input_size=parameters.word_embedding_size,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        """
        :param input_size: embedding后的维度
        :param output_size: 可能的输出类别个数（8）
        :param num_channels:
        :param kernel_size:
        :param dropout:
        :param emb_dropout:
        :param tied_weights:
        """
        super(CpsTcnModel, self).__init__()
        self.tcn = torch.load('../tcn_test_10_2/model/model.pkl')
        for p in self.parameters():
            p.requires_grad = False
        self.crf = CRF(output_size, batch_first=True)

    def forward(self, actions, tags):
        """"""
        feats = self.tcn(actions)
        # print('predict0', feats.argmax(1))
        feats = feats.view(1, -1, parameters.output_size)
        # print('feats', feats.size())

        loss = self.crf(feats, tags)
        loss = - loss
        predict = self.crf.decode(feats)
        # print('predict', predict)
        predict = torch.tensor(predict[0], dtype=torch.int64).to(parameters.device)
        return loss, predict
