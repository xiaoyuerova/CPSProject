### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

序列不足的（开头），用['PAD']补齐

### data:
滑动窗口半径大小 Sliding_window_radius = 2， 序列有5个action

句子最大长度 Sentence_max_length = 16


### model：
使用embeddingBag,做综合模型



基于模型5
将数据分为y_before、y_center、y_behind三份

#### embedding
bert语料库

#### tcn
(1) y_before 和 y_behind 通过 一个tcn模型；
(2) y_center通过另一个tcn模型

#### 线性层：
(1) 768 -> 50;   768 -> 50
(2) 768 -> 50

50 * 3 -> 11


