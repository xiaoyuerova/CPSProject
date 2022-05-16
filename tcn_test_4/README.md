### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

### data:
滑动窗口半径大小 Sliding_window_radius = 4， 序列有8个action的label


### model：
action -> label
训练一个不利用action间序列信息的空白对照组
并用上kappa指数
