import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from tcn_test_0.data_tcn.parameters import Parameters
from tcn_test_0.data_tcn.MyDataset import MyDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
# tokenizer = get_tokenizer('basic_english')
parameters = Parameters()


def label_pipeline(label):
    return int(label)


class Data:
    def __init__(self, dataset: MyDataset, vocab: Vocab):
        """

        :param dataset:
        """
        self.vocab = vocab
        self.text_pipeline = lambda x: self.vocab(tokenizer(x))
        self.label_pipeline = label_pipeline
        self.device = parameters.device
        self.dataloader = DataLoader(dataset, batch_size=parameters.Batch_size, shuffle=False,
                                     collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)
