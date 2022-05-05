import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import logging

logging.set_verbosity_warning()


def f(token_ids, attention_mask):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return bert_model(token_ids, attention_mask)


def data_process(batches: torch.Tensor, max_length):
    batch_size = batches.size()[0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = tokenizer.tokenize(sentence)



class Embedding(nn.Module):
    def __init__(self, max_length=100):
        super(Embedding, self).__init__()
        self.max_length = max_length
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def format(self, x):
        """

        :param x: torch.tenser() [batch_size, sequence_length,sentence_length]
        :return:
        """
        tokens = data_process(x, self.max_length)


