# coding: utf-8
# 2022/4/3 @ tongshiwei

import math
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr
import numpy as np
from KagglePatent import MIRROR, CACHE_DIR
from longling import json_load
from KagglePatent import CLASSMAP, INVERSEMAP, WEIGHT
from datasets import load_dataset as _load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer as _Trainer
from transformers import DataCollatorWithPadding
from KagglePatent.utils import initial_exp, set_devices
from fire import Fire
from baize.metrics import classification_report
from longling import build_dir, as_out_io
from torch.utils.data import WeightedRandomSampler
import torch


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
        if self.imbalance_strategy == "loss":
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                # 计算损失的方法
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                logits = outputs['logits']
                criterion = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(WEIGHT)
                ).to(logits.device)
                loss = criterion(logits, inputs['labels'])

            return (loss, outputs) if return_outputs else loss
        else:
            return super(Trainer, self).compute_loss(model, inputs, return_outputs)


def compute_score(preds, strategy="prob"):
    if strategy == "prob":
        preds = softmax(preds, 1)
        return np.dot(preds, [0, 0.25, 0.5, 0.75, 1]).tolist()
    else:
        return [INVERSEMAP[k] for k in np.argmax(preds, axis=1).tolist()]


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    print(classification_report(labels, np.argmax(logits, axis=1)))
    max_scores = compute_score(logits, "max")
    prob_scores = compute_score(logits, "prob")
    ground_truth = [INVERSEMAP[k] for k in labels]
    return {
        "pearson_correlation_coefficient(max)": pearsonr(ground_truth, max_scores)[0],
        "pearson_correlation_coefficient(prob)": pearsonr(ground_truth, prob_scores)[0]
    }


# 返回测试结果
def back_test(trainer, dataset, with_label=True):
    preds, *_ = trainer.predict(dataset)
    df_data = {
        # dataset id
        "id": dataset["id"],
        "logits": preds.tolist(),
        "softmax": softmax(preds, axis=-1).tolist(),
        "class": compute_score(preds, strategy="max"),
        "score": compute_score(preds, strategy="prob")
    }
    if with_label:
        # 查 INVERSEMAP
        df_data["label"] = [INVERSEMAP[k] for k in dataset["label"]]
        metrics = classification_report(dataset["label"], np.argmax(df_data["logits"], axis=1))
        metrics.update({
            "pearson_correlation_coefficient(max)": pearsonr(df_data["label"], df_data["class"])[0],
            "pearson_correlation_coefficient(prob)": pearsonr(df_data["label"], df_data["score"])[0],
        })
    else:
        metrics = None
    df = pd.DataFrame(df_data)
    return df, metrics


def get_diff_id(df, tolerance=5e-2, key="score"):
    # score, class
    return df[(df[key] - df["label"]).abs() > tolerance]["id"]


# prompt在这
def load_dataset(train_filepath, test_filepath, cpc_filepath, tokenizer):
    template = "In CPC topic {}, the phrases {} and {}"
    cpc = json_load(cpc_filepath)

    data_files = {"test": test_filepath}
    if train_filepath is not None:
        data_files["train"] = train_filepath

    dataset = _load_dataset("csv", data_files=data_files)

    def _templatify(example):
        example["text"] = template.format(cpc[example["context"]], example["anchor"], example["target"])
        example["label"] = CLASSMAP[str(example.get("score", 0.0))]
        example["weight"] = WEIGHT[example["label"]]
        return example

    dataset = dataset.map(
        _templatify,
        remove_columns=["context", "anchor", "target"],
        num_proc=8
    )

    def _tokenize(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = dataset.map(
        _tokenize,
        batched=True,
        num_proc=8
    )

    return dataset


def finetune(train_filepath, valid_filepath, test_filepath, cpc_filepath, model_path, output_dir, model_dir,
             ctx=None, batch_size=16, lr=None, imbalance=None):
    print("model dir is %s" % model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        cache_dir=CACHE_DIR,
        mirror=MIRROR
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    device_num = set_devices(ctx)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=5
    )
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
        train_pred, train_metrics = back_test(trainer, dataset["train"])
        get_diff_id(train_pred).to_csv(model_dir + "wrong_train.csv", index=False)
        get_diff_id(train_pred, key="class").to_csv(model_dir + "wrong_train_class.csv", index=False)
        valid_pred, valid_metrics = back_test(trainer, dataset["test"])
        valid_pred.drop(columns=["label"]).to_csv(model_dir + "valid.csv", index=False)
        get_diff_id(valid_pred).to_csv(model_dir + "wrong_valid.csv", index=False)
        get_diff_id(valid_pred, key="class").to_csv(model_dir + "wrong_valid_class.csv", index=False)

        with as_out_io(model_dir + "metrics.txt") as wf:
            print("******** Train *********", file=wf)
            print(train_metrics, file=wf)
            print("******** Valid *********", file=wf)
            print(valid_metrics, file=wf)

    if test_filepath:
        print("******************* Start Testing ***************")
        dataset = load_dataset(
            None,
            test_filepath,
            cpc_filepath,
            tokenizer
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            eval_accumulation_steps=1,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        build_dir(model_dir, parse_dir=False)
        back_test(trainer, dataset["test"], with_label=False)[0].to_csv(model_dir + "test.csv", index=False)


def main(model="deberta-v3-large", ctx=None, batch_size=16, lr=None, fold=0,
         cat="cv_stratified", mode=None, only_predict=False,
         suffix="", imbalance=None):
    # cv boosting/boosting1
    data_dir = "../../data/"
    if mode != "debug":
        train_path = data_dir + "%s/%s/train.csv" % (cat, fold)
        valid_path = data_dir + "%s/%s/valid.csv" % (cat, fold)
        test_path = data_dir + "test.csv"
        model_dir = data_dir + "{}/%s/%s-classification%s/%s/" % (cat, model, suffix, fold)
    else:
        train_path = data_dir + "debug/train.csv"
        valid_path = data_dir + "debug/valid.csv"
        test_path = data_dir + "debug/valid.csv"
        model_dir = data_dir + "{}/debug/%s-classification%s/" % (model, suffix)

    if only_predict:
        train_path = None
        valid_path = None
        pretrained_model_dir = model_dir.format("model_zoo")
    else:
        pretrained_model_dir = data_dir + "hfl_models/%s" % model

    finetune(
        train_path,
        valid_path,
        test_path,
        data_dir + "cpc_dict.json",
        pretrained_model_dir,
        model_dir.format("tmp"),
        model_dir.format("model_zoo"),
        ctx=ctx,
        batch_size=batch_size,
        lr=lr,
        imbalance=imbalance
    )


if __name__ == '__main__':
    initial_exp()

    Fire(main)
