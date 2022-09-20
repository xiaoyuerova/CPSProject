from transformers import DistilBertTokenizerFast, DistilBertModel
import math
import torch
import pandas as pd
from .MyDataset import MyDataset
from .parameters import parameters
import numpy as np
from sklearn.model_selection import train_test_split


def pretrained_process(data_path):
    df = pd.read_csv(data_path)
    df.reset_index(inplace=True)
    texts = df['Texts']
    labels = df['Labels']
    return texts.tolist(), labels.tolist()


def load_dataset(train_filepath, valid_filepath, cpc_filepath, tokenizer):
    # texts, labels = pretrained_process()
    # temp_outputs = data_split(texts, labels, 0.8, valid_prop=0.9)
    # train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = (item.to_list() for item in
    #                                                                              temp_outputs)
    train_texts, train_labels = pretrained_process(train_filepath)
    val_texts, val_labels = pretrained_process(valid_filepath)

    train_inputs, train_mask, train_labels = data2tensor(train_texts, train_labels, tokenizer)
    val_inputs, val_mask, val_labels = data2tensor(val_texts, val_labels, tokenizer)
    # test_inputs, test_mask, test_labels = data2tensor(test_texts, test_labels, tokenizer)
    return {
        'train': MyDataset(train_inputs, train_mask, train_labels),
        'valid': MyDataset(val_inputs, val_mask, val_labels)
    }


def data2tensor(sentences, labels, tokenizer):
    input_ids, attention_mask = [], []
    # input_ids是每个词对应的索引idx ;token_type_ids是对应的0和1，标识是第几个句子；attention_mask是对句子长度做pad
    # input_ids=[22,21,...499] token_type_ids=[0,0,0,0,1,1,1,1] ;attention_mask=[1,1,1,1,1,0,0,0]补零
    for i in range(len(sentences)):
        encoded_dict = tokenizer.encode_plus(
            sentences[i],
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=16,  # Pad & truncate all sentences.
            padding='max_length',  # 补全操作
            truncation=True,  # 截断操作
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)  # 把多个tensor合并到一起
    attention_mask = torch.cat(attention_mask, dim=0)

    input_ids = torch.LongTensor(input_ids)  # 每个词对应的索引
    attention_mask = torch.LongTensor(attention_mask)  # [11100]padding之后的句子
    labels = torch.LongTensor(labels)  # 所有实例的label对应的索引idx

    return input_ids.to(parameters.device), attention_mask.to(parameters.device), labels.to(parameters.device)


def data_split(data: pd.DataFrame, labels: pd.DataFrame, test_prop, valid_prop=None):
    """
    划分数据集
    :param data:
    :param labels:
    :param test_prop:
    :param valid_prop:
    :return:
    """
    length = len(labels)
    if valid_prop is not None:
        d1 = math.floor(length * test_prop)
        d2 = math.floor(length * valid_prop)
        return (data.loc[:d1], data.loc[d1 + 1:d2], data.loc[d2 + 1:],
                labels.loc[:d1], labels.loc[d1 + 1:d2], labels.loc[d2 + 1:])
    else:
        d1 = math.floor(length * test_prop)
        return data.loc[:d1], data.loc[d1 + 1:], labels.loc[:d1], labels.loc[d1 + 1:]
