import math

import numpy as np
import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import optim
import torch.nn as nn

from tcn_test_6_2.data_tcn import *
from tcn_test_6_2.data_tcn.parameters import Parameters
from tcn_test_6_2.model import CpsTcnModel
from tcn_test_6_2.model2 import CpsTcnModel2
from tcn_test_6_2.utils import build_vocab_from_iterator_re, data_iter

parameters = Parameters()

_dir = '../data/tcn_test_data/tcn-model-data3.csv'
df = pd.read_csv(_dir)
df = df[df['Action_S'].notna()]

# 按组划分测试数据
grouped_data = df.groupby(['NewName'])
divide = int(len(grouped_data) * 0.8)
df_list1 = []
df_list2 = []
for idx, (name, group) in enumerate(grouped_data):
    if idx < divide:
        df_list1.append(group)
    else:
        df_list2.append(group)
df_train = pd.concat(df_list1)
df_test = pd.concat(df_list2)

# 初始化数据
df_train.reset_index(inplace=True)
dataset_train = MyDataset(df_train)
vocab = build_vocab_from_iterator_re(data_iter(dataset_train))
vocab_size = len(vocab)
data_train = Data(dataset_train, vocab)

df_test.reset_index(inplace=True)
dataset_test = MyDataset(df_test)
data_test = Data(dataset_test, vocab)

# 准备模型
model1 = CpsTcnModel(vocab_size, 11, 3)
model1.to(parameters.device)
model2 = CpsTcnModel2(vocab_size, 11, 3)
model2.to(parameters.device)
M = [model1, model2]
# print(model)
total = 0
total2 = 0
for param in model1.parameters():
    total += param.nelement()
    if param.requires_grad:
        total2 += param.nelement()
print("Number of parameter: %.2fM" % (total / 1e6))
print("Number of training parameter: %.2fM" % (total2 / 1e6))

criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=parameters.lr)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=parameters.lr)
optimizer_box = [optimizer1, optimizer2]


def evaluate_ensemble(data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    y1 = []
    y2 = []
    start_time = time.time()
    M[0].eval()
    M[1].eval()
    for idx, (labels, texts, offsets) in enumerate(data.dataloader):
        output1 = M[0](texts, offsets)
        output2 = M[1](texts, offsets)
        output = (output1 + output2) / 2

        total += labels.size(0)

        loss = criterion(output, labels)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(labels):
            if i:
                correct += 1

        y1.extend(predict.to('cpu'))
        y2.extend(labels.to('cpu'))

    batches = data.dataloader.__len__()
    cur_loss = total_loss / batches
    elapsed = time.time() - start_time
    kappa = cohen_kappa_score(y1, y2)
    print(
        '| model ensemble: | epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            epoch + 1, batches,
            elapsed * 1000 / batches, cur_loss,
            correct / total * 100,
            kappa))
    return correct / total


def evaluate(model_index, data: Data, epoch: int):
    optimizer = optimizer_box[model_index]
    total_loss = 0
    correct = 0.0
    total = 0.0
    y1 = []
    y2 = []
    start_time = time.time()
    M[model_index].eval()
    for idx, (labels, texts, offsets) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = M[model_index](texts, offsets)

        total += labels.size(0)

        loss = criterion(output, labels)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(labels):
            if i:
                correct += 1

        y1.extend(predict.to('cpu'))
        y2.extend(labels.to('cpu'))

    batches = data.dataloader.__len__()
    cur_loss = total_loss / batches
    elapsed = time.time() - start_time
    kappa = cohen_kappa_score(y1, y2)
    print(
        '| model{}: | epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            model_index + 1,
            epoch + 1, batches,
            elapsed * 1000 / batches, cur_loss,
            correct / total * 100,
            kappa))
    return correct / total


def train(model_index, train_data: Data, epoch: int):
    optimizer = optimizer_box[model_index]
    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    M[model_index].train()
    # for item in train_data.dataloader:
    #     print(item)
    #     break
    for idx, (labels, texts, offsets) in enumerate(train_data.dataloader):
        optimizer.zero_grad()
        output = M[model_index](texts, offsets)

        total += labels.size(0)
        for i in output.argmax(1).eq(labels):
            if i:
                correct += 1

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        log_interval = parameters.log_interval
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| model{}: | epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | accuracy {:8.2f}%'.format(model_index + 1, epoch + 1, idx,
                                                            train_data.dataloader.__len__(),
                                                            optimizer.param_groups[0]['lr'],
                                                            elapsed * 1000 / log_interval, cur_loss,
                                                            correct / total * 100))
            total_loss = 0
            start_time = time.time()


def main():
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 1.0, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 1.0, gamma=0.1)
    scheduler_box = [scheduler1, scheduler2]
    total_accu1 = None
    total_accu2 = None
    total_accu_box = [total_accu1, total_accu2]
    for epoch in range(parameters.epochs):
        for model_index in [0, 1]:
            train(model_index, data_train, epoch)
            accu_val = evaluate(model_index, data_test, epoch)
            if total_accu_box[model_index] is not None and total_accu_box[model_index] > accu_val:
                scheduler_box[model_index].step()
            else:
                total_accu_box[model_index] = accu_val
        evaluate_ensemble(data_test, epoch)


if __name__ == '__main__':
    main()
