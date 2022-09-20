from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
from transformers import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

data = pd.read_csv('../data.csv')
x = data['Texts'].astype('str')
y = data['Labels']
x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
dataset = {'train': [], 'test': []}
for index in range(len(x_train)):
    input_example = InputExample(text_a=x_train[index], label=int(y_train[index]), guid=index)
    dataset['train'].append(input_example)
for index in range(len(x_test)):
    input_example = InputExample(text_a=x_test[index], label=int(y_test[index]), guid=index)
    dataset['test'].append(input_example)
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
template_text = '{"placeholder":"text_a"} It was {"mask"}.'
myTemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
wrapped_tokenizer = WrapperClass(max_seq_length=16, decoder_max_length=3, tokenizer=tokenizer,
                                 truncate_method="head")
model_inputs = {}
for split in ['train', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_tokenizer.tokenize_one_example(myTemplate.wrap_one_example(sample),
                                                                   teacher_forcing=False)
        model_inputs[split].append(tokenized_example)
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=myTemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=16, decoder_max_length=3,
                                    batch_size=64, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
myVerbalizer = SoftVerbalizer(tokenizer, plm, num_classes=8)
use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=myTemplate, verbalizer=myVerbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()
loss_func = torch.nn.CrossEntropyLoss()
# it's always good practice to set no decay to biase and LayerNorm parameters
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
for epoch in range(10):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
            print('training acc: ',
                  accuracy_score(labels.cpu().data.numpy(), np.argmax(logits.cpu().data.numpy(), axis=1)))
prompt_model.eval()
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=myTemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=16, decoder_max_length=3,
                                   batch_size=64, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")
allPred = []
allLabel = []
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logit = prompt_model(inputs)
    labels = inputs['label']
    allLabel.extend(labels.cpu().tolist())
    allPred.extend(torch.argmax(logit, dim=-1).cpu().tolist())
print('test acc: ', accuracy_score(allPred, allLabel))
print('test kappa: ', cohen_kappa_score(allPred, allLabel))
print(classification_report(allLabel, allPred))
