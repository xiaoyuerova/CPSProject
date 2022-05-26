import torch
from tcn_test_5.data_tcn.parameters import Parameters

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
    tokens_before = tokens[:, :parameters.Sliding_window_radius]
    tokens_center = tokens[:, parameters.Sliding_window_radius]
    tokens_behind = tokens[:, parameters.Sliding_window_radius + 1:]
    return cat_sentence(tokens_before), tokens_center, cat_sentence(tokens_behind)


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
