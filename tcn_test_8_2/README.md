### MyDataset：
首先将数据按 ['NewName', 'LevelNumber'] 分组，以组作为一个序列

### data:
因为序列长度不同，设bz = 1，方便用tensor处理


### model：
使用embeddingBag对句子做嵌入，用tcn进行序列建模。

8-1修改：模型评估，只评估chat数据预测的效果
8-2修改：模型评估，只评估chat数据预测的效果；新增一个tcn识别单句模型；
（想搞一个直接修改embeddingBag参数的通道，但这样应该不对，在8-3中修改）

基于模型6—1—1
使用classification_report评估模型
target_names = ['SESU', 'SMC', 'SSI', 'SN', 'CEU', 'CRF', 'CP', 'CE', 'CM']

基于6-1
考虑时序信息
embeddingBag + tcn，不做数据划分，把模型6简化

基于模型6
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


