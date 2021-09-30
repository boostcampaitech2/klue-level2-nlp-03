import pickle as pickle
import os
import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def split_dataset(self, val_ratio):
    """ 데이터셋을 train 과 val 로 나누기 위한 함수"""
    n_val = int(len(self) * val_ratio)
    n_train = len(self) - n_val
    train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
    return train_set, val_set


def preprocessing_dataset(dataset, rbert=True):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  new_sentence = []

  for sent, i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    # i = i[1:-1].split(',')[0].split(':')[1]
    # j = j[1:-1].split(',')[0].split(':')[1]
    i_word = eval(i)['word'] # eval(): str -> dict
    j_word = eval(j)['word']

    subject_entity.append(i_word)
    object_entity.append(j_word) 

    if rbert:
      i_start = eval(i)['start_idx']
      i_end = eval(i)['end_idx']
      j_start = eval(j)['start_idx']
      j_end = eval(j)['end_idx']

      if i_start < j_start:
        new_sent = sent[:i_start] + '[E11]' + sent[i_start:i_end+1] + '[E12]' + sent[i_end+1:j_start] + '[E21]' + sent[j_start:j_end+1] + '[E22]' + sent[j_end+1:]
      else:
        new_sent = sent[:j_start] + '[E21]' + sent[j_start:j_end+1] + '[E22]' + sent[j_end+1:i_start] + '[E11]' + sent[i_start:i_end+1] + '[E12]' + sent[i_end+1:]
 
      new_sentence.append(new_sent)

  if rbert:
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':new_sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  else:
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, rbert=True):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, rbert)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False,
      )

  return tokenized_sentences