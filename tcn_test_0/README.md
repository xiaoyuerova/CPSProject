### MyDataset：


### data:



### model：
action -> label

baseline

使用的embedingBag
优化器使用SGD
使用了梯度裁剪：torch.nn.utils.clip_grad_norm_
学习率使用scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)，
根据模型再测试集上的效果进行调整。（这里好像应该有验证集，一开始写的时候没有放）
