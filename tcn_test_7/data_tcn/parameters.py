import torch


class Parameters:
    # 滑动窗口大小
    Sliding_window_radius = 4

    # 句子最大长度
    Sentence_max_length = 16

    # 喂数据的batch size
    Batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10

    lr = 0.5

    log_interval = 800

    embedding_size = 256

    # k折, 交叉验证
    n_splits = 10
    random_state = 1
