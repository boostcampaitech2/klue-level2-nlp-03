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

  # args.path 내 모든 csv 파일 읽어오기
  file_list = os.listdir(args.path)
  
  data_list = []
  for csv in file_list:
    data = pd.read_csv(os.path.join(args.path, csv))
    data_list.append(data)

  ensemble_data = [[0]*30] * len(data_list[0])
  data_id = data_list[0]['id']
  
  # csv의 probs 항목 더해서 ensemble_data에 append
  for data in data_list:
    for idx in range(len(data)):
      row = data['probs'][idx]
      row = ast.literal_eval(row)
      ensemble_data[idx] = [x+y for x, y in zip(ensemble_data[idx], row)]
  
  # ensemble_data의 각 row를 csv 파일의 수만큼 나눈 다음, 가장 확률 값이 높은 class 선정.
  preds = []
  for idx, data in enumerate(ensemble_data):
    temp = [x/len(data_list) for x in data]
    ensemble_data[idx] = temp
  
    preds.append(np.argmax(temp))
  
  pred_answer = num_to_label(preds) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  # lists to dataframe, dataframe to csv
  output = pd.DataFrame({'id':data_id,'pred_label':pred_answer,'probs':ensemble_data,})
  output.to_csv(args.path, index=False)
  print(output)
  
def hard_voting(args):
  # args.path 내 모든 csv 파일 읽어오기
  file_list = os.listdir(args.path)
  # hard voting args
  if args.weight_list:
    weight_list = ast.literal_eval(args.weight_list)
  else:
    weight_list = [1] * len(file_list)
  
  assert len(file_list) == len(weight_list)
  
  data_list = []
  for csv in file_list:
    print(csv)
    data = pd.read_csv(os.path.join(args.path, csv))
    data_list.append(data)

    ensemble_data = [[0]*30] * len(data_list[0])
    data_id = data_list[0]['id']

    # csv의 probs 항목 더해서 ensemble_data에 append
    for d_idx, data in enumerate(data_list):
      for idx in range(len(data)):
        row = data['probs'][idx]
        row = ast.literal_eval(row)
        temp = [0]*30
        temp[np.argmax(row)] += 1 * weight_list[d_idx]
        ensemble_data[idx] = [x+y for x, y in zip(ensemble_data[idx], temp)]
  
    # ensemble_data의 각 row를 csv 파일의 수만큼 나눈 다음, 가장 확률 값이 높은 class 선정.
    preds = []
    for idx, data in enumerate(ensemble_data):
      temp = [x/sum(weight_list) for x in data]
      ensemble_data[idx] = temp
  
      preds.append(np.argmax(temp))
  
    pred_answer = num_to_label(preds) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  # lists to dataframe, dataframe to csv
  output = pd.DataFrame({'id':data_id,'pred_label':pred_answer,'probs':ensemble_data,})
  output.to_csv(args.csv_path, index=False)
  print(output)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--path', type=str, default="./prediction_labeled")
  parser.add_argument('--csv_path', type=str, default='submission.csv')
  parser.add_argument('--voting', type=str, default='soft')
  parser.add_argument('--weight_list', type=str)
  args = parser.parse_args()

  if args.voting == 'soft':
    soft_voting(args)
  else:
    hard_voting(args)
