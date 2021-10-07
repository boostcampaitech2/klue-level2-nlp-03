import pandas as pd
import os
import numpy as np
import ast
import argparse
import pickle as pickle

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def soft_voting(args):
  file_list = os.listdir(args.path)
  
  data_list = []
  for csv in file_list:
    data = pd.read_csv(os.path.join(args.path, csv))
    data_list.append(data)

  ensemble_data = [[0]*30] * len(data_list[0])
  data_id = data_list[0]['id']
  
  for data in data_list:
    for idx in range(len(data)):
      row = data['probs'][idx]
      row = ast.literal_eval(row)
      ensemble_data[idx] = [x+y for x, y in zip(ensemble_data[idx], row)]
  
  preds = []
  for idx, data in enumerate(ensemble_data):
    temp = [x/len(data_list) for x in data]
    ensemble_data[idx] = temp
    
    max_value = max(temp)
    max_index = temp.index(max_value)
    preds.append(max_index)
  
  pred_answer = num_to_label(preds) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  output = pd.DataFrame({'id':data_id,'pred_label':pred_answer,'probs':ensemble_data,})
  output.to_csv("ensemble_csv.csv", index=False)
  print(output)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--path', type=str, default="./prediction_labeled")
  args = parser.parse_args()

  soft_voting(args)