import torch


class Parameters:
    # 滑动窗口大小
    Sliding_window_radius = 4

    # 句子最大长度
    Sentence_max_length = 16
    # action序列最大长度
    sequence_max_length = 1024

    # 喂数据的batch size
    Batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10

    lr = 0.5

    log_interval = 20

    word_embedding_size = 64
    sentence_embedding_size = 32

    sentence_output_size = 10000

    target_names = ['SESU', 'SMC', 'SN', 'SSI', 'CRF', 'CP', 'CEC', 'CMC']

    all_names = ['SESU', 'SMC', 'SN', 'SSI', 'CRF', 'CP', 'CE', 'CM', 'CEU', 'CEC', 'CMC']

    pad_num = sentence_output_size + 7

# 5000部分Action
# 0,SESU,3317
# 1,SMC,1292
# 2,SN,1149
# 3,SSI,6177
# 4,CRF,356
# 5,CP,1066
# 6,CE,10
# 7,CM,3
# 8,CEU,3
# 9,CEC,1348
# 10,CMC,1193

# 全部Action
# 0,SESU,3317
# 1,SMC,1292
# 2,SN,1153
# 3,SSI,6182
# 4,CRF,356
# 5,CP,1070
# 6,CE,23582
# 7,CM,5973
# 8,CEU,5312
# 9,CEC,1350
# 10,CMC,1194
