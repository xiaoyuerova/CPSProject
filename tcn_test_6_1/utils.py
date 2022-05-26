import torch
from transformers import BertTokenizer
from tcn_test_6_1.data_tcn.MyDataset import MyDataset
from tcn_test_6_1.data_tcn.parameters import Parameters
from torchtext.vocab import build_vocab_from_iterator


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
parameters = Parameters()


def build_vocab_from_iterator_re(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "[PAD]"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def data_iter(dataset: MyDataset):
    for index in range(0, dataset.__len__()):
        yield dataset.__getitem__(index)


def yield_tokens(d_iter):
    for _, texts in d_iter:
        for text in texts:
            yield tokenizer(text)


def generate_mask(x: torch.Tensor):
    """bertModel要用的mask"""
    mask = []
    for sentence in x:
        temp = [1 if t != 0 else 0 for t in sentence]
        mask.append(temp)
    return torch.tensor(mask, dtype=torch.int64)


def divide_data_for_tcn(tokens: torch.Tensor):
    """
    将数据划分为：中心句前数据，中心句，中心句后数据
    :param tokens:
    :return:
    """
    tokens_before = tokens[:, :parameters.Sliding_window_radius]
    tokens_center = tokens[:, parameters.Sliding_window_radius]
    tokens_behind = tokens[:, parameters.Sliding_window_radius + 1:]
    return tokens_before, tokens_center, tokens_behind


# def cat_sentence(tokens: torch.tensor):
#     """
#
#     :param tokens: [batch_size, n, sentence_length, embed_size]
#     :return:
#     """
#     temp = []
#     for batch in range(tokens.size(0)):
#         t = tokens[batch]
#         temp.append(torch.cat([t[index] for index in range(t.size(0))], dim=0).view(1, -1, 768))
#     return torch.cat(temp, dim=0)
