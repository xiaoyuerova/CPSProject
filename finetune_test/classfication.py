# coding: utf-8


import math
import os

import pandas as pd
from matplotlib.font_manager import json_load
from scipy.special import softmax
from scipy.stats import pearsonr
import numpy as np
# from KagglePatent import MIRROR, CACHE_DIR
# from longling import json_load
# from KagglePatent import CLASSMAP, INVERSEMAP, WEIGHT
from datasets import load_dataset as _load_dataset
from transformers import logging, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer as _Trainer
from transformers import DataCollatorWithPadding
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# from KagglePatent.utils import initial_exp, set_devices
from fire import Fire
# from baize.metrics import classification_report
from sklearn.metrics import classification_report
# from longling import build_dir, as_out_io
from torch.utils.data import WeightedRandomSampler
import torch

from my_datasets import parameters


# compute_score 用。（暂时没有用上）
INVERSEMAP = [i for i in range(8)]

logging.set_verbosity_warning()
logging.set_verbosity_error()


class Trainer(_Trainer):
    def __init__(self, imbalance_strategy=None, *args, **kwargs):
        # sampling, loss
        self.imbalance_strategy = imbalance_strategy
        super(Trainer, self).__init__(*args, **kwargs)

    def _get_train_sampler(self):
        if self.imbalance_strategy == "sampling":
            return WeightedRandomSampler(
                self.train_dataset["weight"],
                len(self.train_dataset)
            )
        else:
            return super(Trainer, self)._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False):
        # 这里inputs是字典
        if self.imbalance_strategy == "loss":
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            print('self.args.past_index', self.args.past_index)
            print(outputs[0])
            print(outputs[1])
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                # 计算损失的方法
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                logits = outputs['logits']
                criterion = torch.nn.CrossEntropyLoss().to(logits.device)
                loss = criterion(logits, inputs['labels'])

            return (loss, outputs) if return_outputs else loss
        else:
            # print("inputs['input_ids']", inputs['input_ids'])
            # print("model(**inputs)['logits'].argmax(dim=1)", model(**inputs)['logits'].argmax(dim=1))
            return super(Trainer, self).compute_loss(model, inputs, return_outputs)


def load_dataset(train_filepath, test_filepath, cpc_filepath, tokenizer):
    # template = "In CPC topic {}, the phrases {} and {}"
    # cpc = json_load(cpc_filepath)

    data_files = {"test": test_filepath}
    if train_filepath is not None:
        data_files["train"] = train_filepath

    dataset = _load_dataset("csv", data_files=data_files)

    # def _templatify(example):
    #     example["text"] = template.format(cpc[example["context"]], example["anchor"], example["target"])
    #     example["label"] = CLASSMAP[str(example.get("score", 0.0))]
    #     example["weight"] = WEIGHT[example["label"]]
    #     return example

    dataset = dataset.map(
        # _templatify,
        remove_columns=["Texts", "Labels"],
        num_proc=8
    )

    def _tokenize(examples):
        return tokenizer(examples["Texts"], truncation=True)

    dataset = dataset.map(
        _tokenize,
        batched=True,
        num_proc=8
    )
    for item in dataset['test']:
        print('item', item)
        break

    return dataset


# 评估的时候用
def compute_score(preds, strategy="prob"):
    if strategy == "prob":
        preds = softmax(preds, 1)
        # 先随便给几个值
        return np.dot(preds, [1, 1, 1, 1, 1, 1, 1, 1]).tolist()
    else:
        print('np.argmax(preds, axis=1).tolist()', np.argmax(preds, axis=1).tolist())
        return [INVERSEMAP[k] for k in np.argmax(preds, axis=1).tolist()]


# 评估的时候用
def compute_metrics(eval_preds):
    # print(eval_preds)
    logits, labels = eval_preds
    print(classification_report(labels, np.argmax(logits, axis=1)))
    max_scores = compute_score(logits, "max")
    prob_scores = compute_score(logits, "prob")
    ground_truth = [INVERSEMAP[k] for k in labels]
    print('ground_truth', ground_truth)
    print('max_scores', max_scores)
    return {
        "pearson_correlation_coefficient(max)": pearsonr(ground_truth, max_scores)[0],
        "pearson_correlation_coefficient(prob)": pearsonr(ground_truth, prob_scores)[0]
    }


def set_devices(devices):
    device_num = 1
    if devices is None:
        return device_num
    if isinstance(devices, tuple):
        device_num = len(devices)
        devices = ",".join(map(str, devices))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    return device_num


def initial_exp(seed=0):
    torch.manual_seed(seed)
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"


def finetune(train_filepath, valid_filepath, test_filepath, cpc_filepath, model_path, output_dir, model_dir,
             ctx=None, batch_size=16, lr=None, imbalance=None):
    """

    :param train_filepath:
    :param valid_filepath:
    :param test_filepath:
    :param cpc_filepath:
    :param model_path:
    :param output_dir:
    :param model_dir:
    :param ctx:
    :param batch_size:
    :param lr:
    :param imbalance:
    :return:
    """
    print("model dir is %s" % model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=16,
        # mirror = MIRROR
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    device_num = set_devices(ctx)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=8)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if train_filepath and valid_filepath:
        print("******************* Start Training ***************")

        dataset = load_dataset(
            train_filepath,
            valid_filepath,
            cpc_filepath,
            tokenizer
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5 * math.sqrt(device_num * batch_size / 16) if lr is None else lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            imbalance_strategy=imbalance
        )

        # 训练模型
        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # 输出测试结果，训练集+测试集
        # train_pred, train_metrics = back_test(trainer, dataset["train"])
        # get_diff_id(train_pred).to_csv(model_dir + "wrong_train.csv", index=False)
        # get_diff_id(train_pred, key="class").to_csv(model_dir + "wrong_train_class.csv", index=False)
        # valid_pred, valid_metrics = back_test(trainer, dataset["test"])
        # valid_pred.drop(columns=["label"]).to_csv(model_dir + "valid.csv", index=False)
        # get_diff_id(valid_pred).to_csv(model_dir + "wrong_valid.csv", index=False)
        # get_diff_id(valid_pred, key="class").to_csv(model_dir + "wrong_valid_class.csv", index=False)

        # with as_out_io(model_dir + "metrics.txt") as wf:
        #     print("******** Train *********", file=wf)
        #     print(train_metrics, file=wf)
        #     print("******** Valid *********", file=wf)
        #     print(valid_metrics, file=wf)

    # if test_filepath:
    #     print("******************* Start Testing ***************")
    #     dataset = load_dataset(
    #         None,
    #         test_filepath,
    #         cpc_filepath,
    #         tokenizer
    #     )
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         per_device_eval_batch_size=batch_size,
    #         eval_accumulation_steps=1,
    #     )
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         tokenizer=tokenizer,
    #         # data_collator=data_collator,
    #     )
    #
    #     build_dir(model_dir, parse_dir=False)
    #     back_test(trainer, dataset["test"], with_label=False)[0].to_csv(model_dir + "test.csv", index=False)


def main(model="deberta-v3-large", ctx=None, batch_size=16, lr=None, fold=0,
         cat="cv_stratified", mode=None, only_predict=False,
         suffix="", imbalance=None):
    # cv boosting/boosting1
    # data_dir = "./data/"
    # if mode != "debug":
    #     train_path = data_dir + "%s/%s/train.csv" % (cat, fold)
    #     valid_path = data_dir + "%s/%s/valid.csv" % (cat, fold)
    #     test_path = data_dir + "test.csv"
    #     model_dir = data_dir + "{}/%s/%s-classification%s/%s/" % (cat, model, suffix, fold)
    # else:
    #     train_path = data_dir + "debug/train.csv"
    #     valid_path = data_dir + "debug/valid.csv"
    #     test_path = data_dir + "debug/valid.csv"
    #     model_dir = data_dir + "{}/debug/%s-classification%s/" % (model, suffix)
    #
    # if only_predict:
    #     train_path = None
    #     valid_path = None
    #     pretrained_model_dir = model_dir.format("model_zoo")
    # else:
    #     # 哈工大讯飞联合实验室(HFL)
    #     pretrained_model_dir = data_dir + "hfl_models/%s" % model

    # print('train_path', train_path)
    # print('model_dir', model_dir)
    # print('pretrained_model_dir', pretrained_model_dir)
    train_path = '../data/single-sentence-prediction/train_data.csv'
    valid_path = '../data/single-sentence-prediction/valid_data.csv'
    test_path = '../data/single-sentence-prediction/test_data.csv'
    pretrained_model_dir = 'distilbert-base-uncased'
    output_dir = './output'
    model_dir = './model'
    ctx = ctx,
    batch_size = batch_size,
    lr = lr,
    imbalance = imbalance
    finetune(
        train_path,
        valid_path,
        test_path,
        './data/cpc/' + "cpc_dict.json",
        pretrained_model_dir,
        output_dir,
        model_dir,
        ctx=ctx,
        batch_size=4,
        lr=5,
        imbalance=imbalance
    )


if __name__ == '__main__':
    initial_exp()
    Fire(main)
