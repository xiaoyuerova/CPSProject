### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

序列不足的（开头），用['PAD']补齐

### data:
滑动窗口半径大小 Sliding_window_radius = 4， 序列有9个action

句子最大长度 Sentence_max_length = 16


### model：
考虑时序信息
模型6-1：embaddingBag + tcn，不做数据划分，把模型6简化
模型0：embaddingBag 单句预测
模型集成

