import copy

import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import optim
import torch.nn as nn

from tcn_test_7.data_tcn import *
from tcn_test_7.data_tcn.parameters import Parameters
from tcn_test_7.model import CpsTcnModel
from tcn_test_7.utils import k_fold_test, get_vocab, generate_model

parameters = Parameters()
vocab = get_vocab()

# 准备模型
# model_base = CpsTcnModel(len(vocab), 11, [parameters.embedding_size] * 3)
# model_base.to(parameters.device)

# 统计模型参数
# total = 0
# total2 = 0
# for param in model_base.parameters():
#     total += param.nelement()
#     if param.requires_grad:
#         total2 += param.nelement()
# print("Number of parameter: %.2fM" % (total / 1e6))
# print("Number of training parameter: %.2fM" % (total2 / 1e6))

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr)


def evaluate(k_step: int, model, data: Data, optimizer):
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
        '| k step {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            k_step + 1, batches,
            elapsed * 1000 / batches, cur_loss,
            correct / total * 100,
            kappa))
    return correct / total, kappa


def train(k_step: int, model, train_data: Data, optimizer):

    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    model.train()
    for idx, (label, text) in enumerate(train_data.dataloader):
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
            print('| k step {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | accuracy {:8.2f}%'.format(k_step + 1, idx, train_data.dataloader.__len__(),
                                                            optimizer.param_groups[0]['lr'],
                                                            elapsed * 1000 / log_interval, cur_loss,
                                                            correct / total * 100))
            total_loss = 0
            start_time = time.time()


def main():
    # scheduler_base = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = [None] * 10
    model = [generate_model(vocab) for idx in range(parameters.n_splits)]
    optimizer = [torch.optim.SGD(model[idx].parameters(), lr=parameters.lr) for idx in range(parameters.n_splits)]
    scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[idx], 1.0, gamma=0.1) for idx in range(parameters.n_splits)]

    accu_vals, kappas = [], []
    for epoch in range(parameters.epochs):
        print('epoch {} '.format(epoch) + '-' * 20)
        for idx, (train_data, test_data) in enumerate(k_fold_test(vocab)):
            train(idx, model[idx], train_data, optimizer[idx])
            accu_val, kappa = evaluate(idx, model[idx], test_data, optimizer[idx])
            accu_vals.append(accu_val)
            kappas.append(kappa)
            if total_accu[idx] is not None and total_accu[idx] > accu_val:
                print('scheduler runs')
                scheduler[idx].step()
            else:
                total_accu[idx] = accu_val
        print('k-fold cross validation: | accuracy {:8.2f}% | Kappa {:8.4f} |'.format(
            sum(accu_vals) / len(accu_vals),
            sum(kappas) / len(kappas),
        ))


if __name__ == '__main__':
    main()
