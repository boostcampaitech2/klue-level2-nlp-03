import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import argparse
import wandb

def train_model_using_knockknock(model, training_args, train_dataset, eval_dataset, compute_metrics, *args, **kwargs):
    # import time
    # time.sleep(10000)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # define metrics function
    )
    return trainer  # Optional return value

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    # train_dataset = load_data("../dataset/train/train.csv")
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
    dataset = load_data("../dataset/train/train.csv")
    # ['sentence','subject_entity','object_entity','label']가 모두 동일한 data 하나만 두고 제거
    dup_data = dataset.duplicated(['sentence', 'subject_entity', 'object_entity', 'label'], keep=False)
    dataset.drop_duplicates(subset=['sentence', 'subject_entity', 'object_entity', 'label'], inplace=True)

    #   # 라벨링 수정
    drop_index = [3296, 25094, 6749, 18458, 8364, 10320, 11511]
    dataset = dataset.drop(index=drop_index, axis=0)

    train_dataset, dev_dataset = train_test_split(dataset, test_size=0.2, shuffle=True, stratify=dataset['label'],
                                                  random_state=42)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        learning_rate=args.learning_rate,  # learning_rate
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,  # log saving step.
        evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=args.eval_steps,  # evaluation step.
        load_best_model_at_end=args.load_best_model_at_end,
        run_name=args.run_name,
    )
    # trainer = Trainer(
    #     model=model,  # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=RE_train_dataset,  # training dataset
    #     eval_dataset=RE_train_dataset,  # evaluation dataset
    #     compute_metrics=compute_metrics  # define metrics function
    # )

    # train model
    train_model_using_knockknock(model=model, training_args=training_args, train_dataset=RE_train_dataset, eval_dataset=RE_dev_dataset, compute_metrics=compute_metrics).train()
    model.save_pretrained(args.output_dir)


def main(args):
    train(args)


if __name__ == '__main__':
    # main()

    parser = argparse.ArgumentParser()

    # args
    parser.add_argument('--run_name', type=str, default="roberta-large_1t")
    parser.add_argument('--model', type=str, default="rbt-l_32_10")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-large")
    parser.add_argument('--output_dir', type=str, default="./roberta-large")
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=int, default=5e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logging_dir', type=str, default="./logs")
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--evaluation_strategy', type=str, default="steps")
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--load_best_model_at_end', type=bool, default=True)

    args = parser.parse_args()

    wandb.init(
        project='KLUE',
        name=args.model,
        config=args,
        entity='bumblebe2'
    )

    # wandb.config.epochs = 4
    wandb.config.update(args)
    main(args)
    print(args)
    # main()
