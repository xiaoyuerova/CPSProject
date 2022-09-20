### MyDataset：


### data:



### model：

序列信息对比实验


使用classification_report评估模型
target_names = ['SESU', 'SMC', 'SN', 'SSI', 'CRF', 'CP', 'CEC', 'CMC']
删除了
6,CE,10
7,CM,3
8,CEU,3

因为
5000部分Action
0,SESU,3317
1,SMC,1292
2,SN,1149
3,SSI,6177
4,CRF,356
5,CP,1066
6,CE,10
7,CM,3
8,CEU,3
9,CEC,1348
10,CMC,1193



基于4-1-1
action -> label
训练一个不利用action间序列信息的空白对照组
并用上kappa指数

基于模型4，不使用bertModel，自己训练embedding
使用分词细化（数据替换）后的数据

在模型4_1 （只用了tcn）的基础上，尝试使用SGD优化器，动态调整lr


#### 错误记录
梯度直接变为nan
可能得原因是初始lr太大
