import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import BertTokenizer
from tcn_test_10_2.data_tcn.parameters import Parameters
from tcn_test_10_2.data_tcn.MyDataset import MyDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
parameters = Parameters()


def text_tokenizer(sentence):
    if type(sentence) != str:
        print('type error', sentence)
    tokens = tokenizer.tokenize(sentence)
    length = len(tokens)
    if length < parameters.Sentence_max_length - 2:
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens = tokens + ['[PAD]'] * (parameters.Sentence_max_length - length - 2)
    else:
        tokens = tokens[:parameters.Sentence_max_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
    return tokens


def label_pipeline(labels):
    return [int(label) for label in labels]


def action_pipeline(sequence, vocab: Vocab):
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
            tokens = tokens[:parameters.Sentence_max_length - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = vocab(tokens)
        sequence_ids.append(tokens_ids)
    return sequence_ids


class Data:
    def __init__(self, dataset: MyDataset, vocab: Vocab):
        """

        :param dataset:
        :return label_list: torch.Tensor
        :return text_list: [int | torch.Tensor]
        """
        self.vocab = vocab
        self.action_pipeline = action_pipeline
        self.label_pipeline = label_pipeline
        self.device = parameters.device
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                     collate_fn=self.collate_batch, drop_last=True)

    def collate_batch(self, batch):
        label_list, action_list = [], []
        for (labels, actions) in batch:
            label_list.append(self.label_pipeline(labels))
            action_list.append(self.action_pipeline(actions, self.vocab))
        labels = torch.tensor(label_list, dtype=torch.int64).to(parameters.device)
        actions = torch.tensor(action_list, dtype=torch.int64).to(parameters.device)
        return labels, actions
