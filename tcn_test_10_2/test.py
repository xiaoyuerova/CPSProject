from transformers import BertTokenizer

from tcn_test_4_1_2.tcn_model_test import main as tcn_model
import torch

if __name__ == '__main__':
    # model = tcn_model()
    # torch.save(model, './model/model.pkl')  # 保存整个模型
    # new_model = torch.load('./model/model.pkl')  # 加载模型
    model = torch.load('../tcn_test_4_1_3/model/model.pkl')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = torch.load('../tcn_test_4_1_3/model/vocab.pkl')
    text = 'what volts does it ask you for?', 'what volts does it ask you for?'
    text = tokenizer.tokenize(text)
    print(text)
    text = vocab(text)
    print(text)
    output = model(text)
    print(output.argmax(1))
