from tcn_test_4_1_2.tcn_model_test import main as tcn_model
import torch

if __name__ == '__main__':
    model = tcn_model()
    torch.save(model, './model/model.pkl')  # 保存整个模型
    new_model = torch.load('./model/model.pkl')  # 加载模型
