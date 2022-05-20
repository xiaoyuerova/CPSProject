### MyDataset：
获取所有chat

### data
一个个Action（只有chat）

### model：
action -> label
训练一个不利用action间序列信息的空白对照组
并用上kappa指数

更换数据，使用data/tcn_test_data/tcn-model-data3.csv，用于与tcn_test_4_1的比较
对bert模型添加special_tokens
