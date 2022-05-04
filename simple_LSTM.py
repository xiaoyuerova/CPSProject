import torch.nn as nn
import torch

device = "cpu"


def train_val_test(model, optimizer, criterion, train_iter, val_iter, test_iter, epochs):
    for epoch in range(1, epochs + 1):
        train_loss = 0.0  # 训练损失
        val_loss = 0.0  # 验证损失
        model.train()  # 声明开始训练
        for indices, batch in enumerate(train_iter):
            optimizer.zero_grad()  # 梯度置0
            outputs = model(batch.dialogue)  # 预测后输出 outputs shape :  torch.Size([32, 2])
            # batch.label shape :  torch.Size([32])
            loss = criterion(outputs, batch.label.long())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # batch.dialogue shape :  torch.Size([26, 32]) --> 26:序列长度， 32:一个batch_size的大小
            train_loss += loss.data.item() * batch.dialogue.size(0)  # 累计每一批的损失值
        train_loss /= len(train_iter)  # 计算平均损失 len(train_iter) :  40000
        print("Epoch : {}, Train Loss : {:.2f}".format(epoch, train_loss))

        model.eval()  # 声明模型验证
        for indices, batch in enumerate(val_iter):
            context = batch.dialogue.to(device)  # 部署到device上
            target = batch.label.to(device)
            pred = model(context)  # 模型预测
            loss = criterion(pred, target)  # 计算损失 len(val_iter) :  5000
            val_loss += loss.item() * context.size(0)  # 累计每一批的损失值
        val_loss /= len(val_iter)  # 计算平均损失
        print("Epoch : {}, Val Loss : {:.2f}".format(epoch, val_loss))

        model.eval()  # 声明
        correct = 0.0  # 计算正确率
        test_loss = 0.0  # 测试损失
        with torch.no_grad():  # 不进行梯度计算
            for idx, batch in enumerate(test_iter):
                context = batch.dialogue.to(device)  # 部署到device上
                target = batch.label.to(device)
                outputs = model(context)  # 输出
                loss = criterion(outputs, target)  # 计算损失
                test_loss += loss.item() * context.size(0)  # 累计每一批的损失值
                # 获取最大预测值索引
                preds = outputs.argmax(1)
                # 累计正确数
                correct += preds.eq(target.view_as(preds)).sum().item()
            test_loss /= len(test_iter)  # 平均损失 len(test_iter) :  5000
            print("Epoch : {}, Test Loss : {:.2f}".format(epoch, test_loss))
            # print("Accuracy : {}".format(100 * correct / (len(test_iter) * batch.dialogue.size(1))))


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(SimpleLSTM, self).__init__()  # 调用父类的构造方法
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # vocab_size词汇表大小， embedding_dim词嵌入维度
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 20)  # 全连接层

    def forward(self, seq):
        seq, _ = seq
        output, (hidden, cell) = self.encoder(self.embedding(seq))
        # output :  torch.Size([24, 32, 100])
        # hidden :  torch.Size([1, 32, 100])
        # cell :  torch.Size([1, 32, 100])
        preds = self.predictor(hidden.squeeze(0))
        return preds
