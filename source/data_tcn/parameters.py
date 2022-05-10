import torch


class Parameters:
    # 滑动窗口大小
    Sliding_window_radius = 2

    # 句子最大长度
    Sentence_max_length = 16

    # 喂数据的batch size
    Batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches = 5

    lr = 2e-3

