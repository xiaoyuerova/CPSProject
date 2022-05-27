### MyDataset：


### data:



### model：
使用k折交叉验证
注意一点：这里的vocab由全体数据统计获得

基于模型4-1-1
action -> label
训练一个不利用action间序列信息的空白对照组
并用上kappa指数

基于模型4，不使用bertModel，自己训练embedding
使用分词细化（数据替换）后的数据

在模型4_1 （只用了tcn）的基础上，尝试使用SGD优化器，动态调整lr


#### 错误记录
梯度直接变为nan
可能得原因是初始lr太大
