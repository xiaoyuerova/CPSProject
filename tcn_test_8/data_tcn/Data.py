import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchtext.vocab import Vocab
from tcn_test_8.data_tcn.parameters import Parameters
from tcn_test_8.data_tcn.MyDataset import MyDataset

parameters = Parameters()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize


def text_tokenizer(sentence):
    tokens = tokenizer(sentence)
    try:
        if len(tokens) > parameters.Sentence_max_length:
            return tokens[:parameters.Sentence_max_length]
        else:
            return tokens
    except:
        print('text_tokenizer wrong')


class Data:
    def __init__(self, dataset: MyDataset, vocab: Vocab):
        """

        :param dataset:
        """
        self.vocab = vocab
        self.text_pipeline = lambda x: self.vocab(text_tokenizer(x))
        self.label_pipeline = lambda x: x.to_list()
        self.device = parameters.device
        self.dataloader = DataLoader(dataset, batch_size=parameters.Batch_size, shuffle=False,
                                     collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        """

        :param batch:
        :return: (
            label_batches: torch [bz]
            text_batches: [torch] [bz, sentence * window]
            offset_batches: torch [bz, window]
        """
        label_batches, text_batches, offset_batches = [], [], []
        for idx, (labels, texts) in enumerate(batch):
            label_batches.append(self.label_pipeline(labels))

            text_list, offset_list = [], [0]
            try:
                for i in texts.index:
                    processed_text = self.text_pipeline(texts[i])
                    text_list.extend(processed_text)
                    offset_list.append(len(processed_text))
                offset_list = offset_list[:-1]
            except:
                print('Data error!', texts)

            offset_batches.append(offset_list)
            text_batches.append(text_list)

        offset_batches = torch.tensor(offset_batches, dtype=torch.int64).cumsum(dim=1)
        text_batches = torch.tensor(text_batches, dtype=torch.int64)
        label_batches = torch.tensor(label_batches, dtype=torch.int64)
        return label_batches.to(self.device), text_batches.to(self.device), offset_batches.to(self.device)
