import os

import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import optim
import torch.nn as nn

from tcn_test_10.data_tcn import *
from tcn_test_10.data_tcn.parameters import Parameters
from tcn_test_10.model import CpsTcnModel
from tcn_test_10.utils import build_vocab_from_iterator_re, data_iter

from transformers import BertModel, BertTokenizer, logging
from sklearn.metrics import classification_report

logging.set_verbosity_warning()
logging.set_verbosity_error()
parameters = Parameters()
torch.autograd.set_detect_anomaly(True)

data_path = os.path.join(os.path.dirname(__file__), '../data/group-random-data/')
train_path = data_path + 'train.csv'
test_path = data_path + 'test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 初始化数据
# df_train.reset_index(inplace=True)
dataset_train = MyDataset(df_train)
vocab = build_vocab_from_iterator_re(data_iter(dataset_train))
vocab_size = len(vocab)
data_train = Data(dataset_train, vocab)

# df_test.reset_index(inplace=True)
dataset_test = MyDataset(df_test)
data_test = Data(dataset_test, vocab)

# 准备模型
model = CpsTcnModel(vocab_size, 8, [parameters.sentence_embedding_size] * 3)
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
    for idx, (label, text) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = model(text)

        label = torch.cat(label)
        output = torch.cat(output)
        # print('output', output.size())

        total += label.size(0)

        loss = criterion(output, label)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(label):
            if i:
                correct += 1

        y_pred.extend(predict.to('cpu'))
        y_true.extend(label.to('cpu'))

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
    print(classification_report(y_true, y_pred, target_names=parameters.target_names))
    return correct / total


def train(train_data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    model.train()
    for idx, (label, sentence) in enumerate(train_data.dataloader):
        optimizer.zero_grad()
        output = model(sentence)
        # print(len(output[0]), len(label[0]))

        label = torch.cat(label)
        output = torch.cat(output)
        # print('output', output.size())

        total += label.size(0)

        # print(len(output), len(label))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        for j in output.argmax(1).eq(label):
            if j:
                correct += 1

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
    return model


if __name__ == '__main__':
    main()
