这里有两个模型CpsTcnModel和CpsTcnModel2.
CpsTcnModel跑不了，因为它对应的data和MyDataset被改掉了。目前文件里的data和Dataset是对应CpsTcnModel2的，CpsTcnModel2可以运行。

#CpsTcnModel

### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

### data:
滑动窗口半径大小 Sliding_window_radius = 8， 序列有16个action（不包含要预测action）

句子最大长度 Sentence_max_length = 20


### model：
将序列数据的每个action纵向压缩为一个embedding单元（word）。用平均值压缩
输入tcn的n是action的数量。
[?,  n,  sentence_n,  k]. mean(dim=2)  ->  [?,  n,  k]


#CpsTcnModel2

### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，在组内按窗口依次获取序列数据。（序列数据：连续且以chat为结尾的序列）

### data:
滑动窗口半径大小 Sliding_window_radius = 2， 序列有5个action（包含要预测action）

句子最大长度 Sentence_max_length = 16


### model：
将序列数据的每个action横向压缩为一个action。用平均值压缩
输入tcn的n是action(sentence)的长度。
[?, n, sentence_n, k].mean(dim=1)  ->  [?, sentence_n, k]
