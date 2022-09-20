from transformers import BertTokenizer
import torch

if __name__ == '__main__':
    # model = tcn_model()
    # torch.save(model, './model/model.pkl')  # 保存整个模型
    # new_model = torch.load('./model/model.pkl')  # 加载模型
    model = torch.load('../tcn_test_10_2/model/model.pkl')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = torch.load('../tcn_test_10_2/model/vocab.pkl')
    text = ['what volts does it ask you for?', 'what volts does it ask you for?']
    text = tokenizer.tokenize(text)
    print(text)
    text = vocab(text)
    print(text)
    output = model(text)
    print(output.argmax(1))
