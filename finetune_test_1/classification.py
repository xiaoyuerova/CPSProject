# coding: utf-8
# 2022/6/23 @ tongshiwei

import os
import math
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr
import numpy as np
from datasets import load_dataset as _load_dataset
from longling import build_dir
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from fire import Fire
from baize.metrics import classification_report
import torch

INVERSEMAP = [i for i in range(8)]


def initial_exp(seed=0):
    torch.manual_seed(seed)
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"


def set_devices(devices):
    device_num = 1
    if devices is None:
        return device_num
    if isinstance(devices, tuple):
        device_num = len(devices)
        devices = ",".join(map(str, devices))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    return device_num


def compute_score(preds, strategy="prob"):
    if strategy == "prob":
        preds = softmax(preds, 1)
        return np.dot(preds, [0, 0.25, 0.5, 0.75, 1]).tolist()
    else:
        return [INVERSEMAP[k] for k in np.argmax(preds, axis=1).tolist()]


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    metrics = classification_report(labels, np.argmax(logits, axis=1))
    print(metrics)
    return {
        "macro_f1": metrics["macro_avg"]["f1"],
    }


def load_dataset(train_filepath, test_filepath, tokenizer):

    data_files = {"test": test_filepath}
    if train_filepath is not None:
        data_files["train"] = train_filepath

    dataset = _load_dataset("csv", data_files=data_files)

    def _tokenize(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = dataset.map(
        _tokenize,
        batched=True,
        num_proc=1
    )

    return dataset


def back_test(trainer, dataset, with_label=True):
    preds, *_ = trainer.predict(dataset)
    df_data = {
        "id": dataset["id"],
        "logits": preds.tolist(),
        "softmax": softmax(preds, axis=-1).tolist(),
        "class": compute_score(preds, strategy="max"),
        "score": compute_score(preds, strategy="prob")
    }
    if with_label:
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


def finetune(train_filepath, valid_filepath, test_filepath, model_path, output_dir, model_dir,
             ctx=None, batch_size=16, lr=None):
    print("model dir is %s" % model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=24,
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    device_num = set_devices(ctx)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=8
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if train_filepath and valid_filepath:
        print("******************* Start Training ***************")

        dataset = load_dataset(
            train_filepath,
            valid_filepath,
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
        )

        trainer.train()
        # model.save_pretrained(model_dir)
        # tokenizer.save_pretrained(model_dir)

    # if test_filepath:
    #     print("******************* Start Testing ***************")
    #     dataset = load_dataset(
    #         None,
    #         test_filepath,
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
    #         data_collator=data_collator,
    #     )
    #
    #     build_dir(model_dir, parse_dir=False)
    #     back_test(trainer, dataset["test"], with_label=False)[0].to_csv(model_dir + "test.csv", index=False)


def main(model="distilroberta-base", ctx=None, batch_size=16, lr=None,
         suffix=""):
    data_dir = "data/"
    train_path = data_dir + "train.csv"
    valid_path = data_dir + "valid.csv"
    test_path = data_dir + "test.csv"
    model_dir = data_dir + "{}/%s-classification%s/" % (model, suffix)
    pretrained_model_dir = data_dir + "hfl_models/%s" % model

    print("train: %s, valid: %s" % (train_path, valid_path))
    print("pretrained model is located at %s" % pretrained_model_dir)

    finetune(
        train_path,
        valid_path,
        test_path,
        pretrained_model_dir,
        model_dir.format("tmp"),
        model_dir.format("model_zoo"),
        ctx=ctx,
        batch_size=batch_size,
        lr=lr,
    )


if __name__ == '__main__':
    initial_exp()

    Fire(main)
