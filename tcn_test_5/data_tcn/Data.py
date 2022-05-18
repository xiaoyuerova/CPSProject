import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tcn_test_5.data_tcn.parameters import Parameters

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
parameters = Parameters()


def text_pipeline(sequence):
    sequence_ids = []
    for sentence in sequence:
        if type(sentence) != str:
            print('type error', sentence)
        tokens = tokenizer.tokenize(sentence)
        length = len(tokens)
        if length < parameters.Sentence_max_length - 2:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            tokens = tokens + ['[PAD]'] * (parameters.Sentence_max_length - length - 2)
        else:
            tokens = tokens[:parameters.Sentence_max_length-2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        sequence_ids.append(tokens_ids)
    return sequence_ids


def label_pipeline(label):
    return int(label)
    # return [int(label) for label in labels]


class Data:
    def __init__(self, dataset):
        """

        :param dataset:
        """
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline
        self.device = parameters.device
        self.dataloader = DataLoader(dataset, batch_size=parameters.Batch_size, shuffle=False, collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _texts) in batch:
            label_list.append(self.label_pipeline(_label))
            text_list.append(self.text_pipeline(_texts))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        return label_list.to(self.device), text_list.to(self.device)

