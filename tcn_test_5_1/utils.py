import torch
from tcn_test_5_1.data_tcn.parameters import Parameters

parameters = Parameters()


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
    tokens_before = tokens[:, 0]
    tokens_center = tokens[:, 1]
    tokens_behind = tokens[:, 2]
    return tokens_before, tokens_center, tokens_behind


def to_tenser(x: [torch.Tensor], index: int):
    batch_size = len(x)
    xx = []
    for i in range(batch_size):
        xx.append(x[i][index])
    return torch.cat([item.view(1, -1) for item in xx], dim=0)


def cat_sentence(tokens: torch.tensor):
    """

    :param tokens: [batch_size, n, sentence_length, embed_size]
    :return:
    """
    temp = []
    for batch in range(tokens.size(0)):
        t = tokens[batch]
        temp.append(torch.cat([t[index] for index in range(t.size(0))], dim=0).view(1, -1, 768))
    return torch.cat(temp, dim=0)
