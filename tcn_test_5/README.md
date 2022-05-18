### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

序列不足的（开头），用['PAD']补齐

### data:
滑动窗口半径大小 Sliding_window_radius = 2， 序列有5个action

句子最大长度 Sentence_max_length = 16


### model：
内部含多个模型


输入tcn的n是action的长度 * 5。
[?,  n,  sentence_n,  k]  ->  [?,  n*sentence_n,  k]
