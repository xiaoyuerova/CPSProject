import math

import numpy as np
import time
import pandas as pd
import torch
from torch import optim
import torch.nn as nn

from tcn_test_2.data_tcn_2 import *
from tcn_test_2.data_tcn_2.parameters import Parameters
from tcn_test_2.model import CpsTcnModel

from transformers import BertModel, BertTokenizer, logging

logging.set_verbosity_warning()
logging.set_verbosity_error()
parameters = Parameters()

model = CpsTcnModel(768, 11, [768] * 4)
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
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(parameters.device)


def generate_mask(x: torch.Tensor):
    mask = []
    for sentence in x:
        temp = [1 if t != 0 else 0 for t in sentence]
        mask.append(temp)
    return torch.tensor(mask, dtype=torch.int64)


def train(data: Data, epoch: int):
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

        log_interval = 500
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f} | accuracy {:8.2f}%'.format(epoch, idx, data.dataloader.__len__(),
                                                                          parameters.lr,
                                                                          elapsed * 1000 / log_interval, cur_loss,
                                                                          math.exp(cur_loss),
                                                                          correct / total * 100))
            total_loss = 0
            start_time = time.time()


def main():
    _dir = '../data/tcn_test_data/tcn-model-data2.csv'
    df = pd.read_csv(_dir)
    df = df[df['Action'].notna()]
    df.reset_index(inplace=True)
    dataset = MyDataset(df)
    data = Data(dataset)

    for epoch in range(parameters.epochs):
        train(data, epoch)


if __name__ == '__main__':
    main()
