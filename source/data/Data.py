import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from source.data import MyDataset
from transformers import BertTokenizer

tokenizer = get_tokenizer('basic_english')
model_transformer = BertTokenizer.from_pretrained('bert-base-uncased')


def yield_tokens(data_iter):
    for _, text in data_iter:
        if type(text) == float:
            print(text)
            continue
        else:
            yield model_transformer.tokenize(text)
            # yield tokenizer(text)


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def data_iter(dataset: MyDataset):
    for index in range(0, dataset.__len__()):
        yield dataset.__getitem__(index)


class Data:
    def __init__(self, dataset):
        """

        :param dataset:
        """
        self.vocab = build_vocab_from_iterator_re(data_iter(dataset))
        self.text_pipeline = lambda x: self.vocab(model_transformer.tokenize(x))
        self.label_pipeline = lambda x: int(x)        # 千万注意要不要-1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=self.collate_batch)

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
