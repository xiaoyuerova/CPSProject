import math

import numpy as np
import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, classification_report
from torch import optim
import torch.nn as nn

from tcn_test_6_1_1.data_tcn import *
from tcn_test_6_1_1.data_tcn.parameters import Parameters
from tcn_test_6_1_1.model import CpsTcnModel
from tcn_test_6_1_1.utils import build_vocab_from_iterator_re, data_iter

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

df_target = df_test[df_test['DataCode'] == 5000]
target_names = []
for i in range(11):
    for index in df_target.index:
        if i == df_target.at[index, 'Label']:
            target_names.append(df_target.at[index, 'Jess0'])
            break

# 准备模型
model = CpsTcnModel(vocab_size, 11, 3)
model.to(parameters.device)
# print(model)
total = 0
total2 = 0
for param in model.parameters():
    total += param.nelement()
    if param.requires_grad:
        total2 += param.nelement()
print("Number of parameter: %.2fM" % (total / 1e6))
print("Number of training parameter: %.2fM" % (total2 / 1e6))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr)


def evaluate(data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    y_pred, y_true = [], []
    start_time = time.time()
    model.eval()
    for idx, (labels, texts, offsets) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = model(texts, offsets)

        total += labels.size(0)

        loss = criterion(output, labels)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(labels):
            if i:
                correct += 1

        y_pred.extend(predict.to('cpu'))
        y_true.extend(labels.to('cpu'))

    batches = data.dataloader.__len__()
    cur_loss = total_loss / batches
    elapsed = time.time() - start_time
    kappa = cohen_kappa_score(y_pred, y_true)
    print(
        '| epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            epoch + 1, batches,
            elapsed * 1000 / batches, cur_loss,
            correct / total * 100,
            kappa))
    print(classification_report(y_true, y_pred, target_names=target_names))
    return correct / total


def train(train_data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    model.train()
    # for item in train_data.dataloader:
    #     print(item)
    #     break
    for idx, (labels, texts, offsets) in enumerate(train_data.dataloader):
        optimizer.zero_grad()
        output = model(texts, offsets)

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
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | accuracy {:8.2f}%'.format(epoch + 1, idx, train_data.dataloader.__len__(),
                                                            optimizer.param_groups[0]['lr'],
                                                            elapsed * 1000 / log_interval, cur_loss,
                                                            correct / total * 100))
            total_loss = 0
            start_time = time.time()


def main():
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    for epoch in range(parameters.epochs):
        train(data_train, epoch)
        accu_val = evaluate(data_test, epoch)
        if total_accu is not None and total_accu > accu_val:
            print('scheduler runs')
            scheduler.step()
        else:
            total_accu = accu_val


if __name__ == '__main__':
    main()
