import torch


class Parameters:
    # 滑动窗口大小
    Sliding_window_radius = 4

    # 句子最大长度
    Sentence_max_length = 16

    # 喂数据的batch size
    Batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 5

    lr = 2e-3

    log_interval = 500

    special_tokens = {'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                    '[voltage]', '[current]', '[number]']}
