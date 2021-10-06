from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from load_data import *

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
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label


# TODO: load your tokenizer & dataset
MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load dataset
train_dataset = load_data("../dataset/train/train.csv", rbert=True)
  
train_label = label_to_num(train_dataset['label'].values)

# tokenizing dataset
tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  

# make dataset for pytorch.
RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  

RE_train_dataset, RE_dev_dataset = RE_train_dataset.split_dataset(0.1)

# TODO: change your pretrained model path
config = AutoConfig.from_pretrained(MODEL_NAME)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=5,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to="wandb",
    run_name="roberta"
  )
  
trainer = Trainer(
    model_init=model_init,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 4, 8, 16, 32, 64),
    }

# NOTE: ray tune
def ray_hp_space(trial):
    from ray import tune
    return {
        "learning_rate": tune.loguniform(5e-6, 5e-4),
        "num_train_epochs": tune.choice(range(1, 6)),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }

best_run = trainer.hyperparameter_search(
    direction="maximize", # NOTE: or direction="minimize"
    hp_space=ray_hp_space, # NOTE: if you wanna use optuna, change it to optuna_hp_space
    backend="ray", # NOTE: if you wanna use optuna, remove this argument
)

for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()