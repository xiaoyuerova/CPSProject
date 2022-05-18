import math

import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import optim
import torch.nn as nn

from tcn_test_4_0.data_tcn import *
from tcn_test_4_0.data_tcn.parameters import Parameters
from tcn_test_4_0.model import CpsTcnModel

from transformers import BertModel, BertTokenizer, logging

logging.set_verbosity_warning()
logging.set_verbosity_error()
parameters = Parameters()

model = CpsTcnModel(768, 11, [768] * 3)
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
optimizer = getattr(optim, 'Adam')(model.parameters(), lr=parameters.lr)


def evaluate(data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    y1 = []
    y2 = []
    start_time = time.time()
    model.eval()
    for idx, (label, text) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = model(text)

        total += label.size(0)

        loss = criterion(output, label)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(label):
            if i:
                correct += 1

        y1.extend(predict.to('cpu'))
        y2.extend(label.to('cpu'))

    batches = data.dataloader.__len__()
    cur_loss = total_loss / batches
    elapsed = time.time() - start_time
    kappa = cohen_kappa_score(y1, y2)
    print(
        '| epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'ppl {:8.2f} | accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            epoch+1, batches,
            elapsed * 1000 / batches, cur_loss,
            math.exp(cur_loss),
            correct / total * 100,
            kappa))


def train(data: Data, test_data: Data, epoch: int):
    steps = 0
    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    model.train()
    for idx, (label, text) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = model(text)

        total += label.size(0)
        for i in output.argmax(1).eq(label):
            if i:
                correct += 1

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        log_interval = parameters.log_interval
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f} | accuracy {:8.2f}%'.format(epoch+1, idx, data.dataloader.__len__(),
                                                                          parameters.lr,
                                                                          elapsed * 1000 / log_interval, cur_loss,
                                                                          math.exp(cur_loss),
                                                                          correct / total * 100))
            total_loss = 0
            start_time = time.time()
    evaluate(test_data, epoch)


def main():
    _dir = '../data/tcn_test_data/tcn-model-data3.csv'
    df = pd.read_csv(_dir)
    df = df[df['DataCode'] == 5000]
    df = df[df['Action_S'].notna()]

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

    df_train.reset_index(inplace=True)
    dataset_train = MyDataset(df_train)
    data_train = Data(dataset_train)

    df_test.reset_index(inplace=True)
    dataset_test = MyDataset(df_test)
    data_test = Data(dataset_test)

    for epoch in range(parameters.epochs):
        train(data_train, data_test, epoch)


if __name__ == '__main__':
    main()
