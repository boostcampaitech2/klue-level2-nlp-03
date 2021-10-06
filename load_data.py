import pandas as pd
import torch

from transformers import BertTokenizer

import itertools

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels
    # self.entity_mask = self.get_entity_embeddings()

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

class RE_rbet_Dataset(RE_Dataset):
  """ R-Roberta의 Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.entity_mask = self.get_entity_embeddings()

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['e1_mask'] = torch.tensor(self.entity_mask[idx][0])
    item['e2_mask'] = torch.tensor(self.entity_mask[idx][1])
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def get_entity_embeddings(self):
    """문장 내 entity의 위치 표시하는 entity_embeddings를 리턴하는 함수"""
    # Needs Refactoring!!!
    entity_list = []
    sent_list = self.pair_dataset['input_ids'].tolist()
    for sent in sent_list:
      first_entity_token = [list(y) for x, y in itertools.groupby(sent, lambda z: z == 36) if not x] # @ == 36 / # == 7
      second_entity_token = [list(y) for x, y in itertools.groupby(sent, lambda z: z == 7) if not x]

      e1_mask = [0] * len(first_entity_token[0]) + [0] 
      e2_mask = [0] * len(second_entity_token[0]) + [0] 

      e1_mask += [1] * len(first_entity_token[1]) + [0]
      e2_mask += [1] * len(second_entity_token[1]) + [0]

      e1_mask += [0] * len(first_entity_token[2])
      e2_mask += [0] * len(second_entity_token[2])

      entity_list.append([e1_mask, e2_mask])

    return entity_list


def preprocessing_dataset(dataset, rbert):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  subject_tag = []
  object_tag = []
  new_sentence = []

  for sent, i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    # i = i[1:-1].split(',')[0].split(':')[1]
    # j = j[1:-1].split(',')[0].split(':')[1]
    i_word = eval(i)['word'] # eval(): str -> dict
    j_word = eval(j)['word']

    i_tag = eval(i)['type']
    j_tag = eval(j)['type']

    subject_entity.append(i_word)
    object_entity.append(j_word) 

    subject_tag.append(i_tag)
    object_tag.append(j_tag)

    if rbert:
      i_start = eval(i)['start_idx']
      i_end = eval(i)['end_idx']
      j_start = eval(j)['start_idx']
      j_end = eval(j)['end_idx']

      if i_start < j_start:
        new_sent = sent[:i_start] + '@' + sent[i_start:i_end+1] + '@' + sent[i_end+1:j_start] + '#' + sent[j_start:j_end+1] + '#' + sent[j_end+1:]
      else:
        new_sent = sent[:j_start] + '#' + sent[j_start:j_end+1] + '#' + sent[j_end+1:i_start] + '@' + sent[i_start:i_end+1] + '@' + sent[i_end+1:]
 
      new_sentence.append(new_sent)

  if rbert:
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':new_sentence,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  else:
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  return out_dataset

def load_data(dataset_dir, rbert):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset, rbert)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  concat_tag = []
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

def custom_tokenized_dataset(dataset):
  """Mecab tokenizer를 사용하는 dataset"""

  vocab_path = './after_mecab_test.txt'
  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
  
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