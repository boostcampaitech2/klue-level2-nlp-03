from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import random
import tqdm

from koeda import EDA
from koeda import AEDA

from load_data import *

def main():
    train_dataset = pd.read_csv("../dataset/train/train.csv")

    #train_dataset 내 중복 문장 제거
    train_dataset = train_dataset[~train_dataset.duplicated(subset=['sentence', 'subject_entity', 'object_entity', 'label'])]
    train_dataset.to_csv("../dataset/train/train_drop_duplicated.csv", index=False)

    train_df = pd.read_csv("../dataset/train/train_drop_duplicated.csv")

    #label 비율 계산
    c = train_df['label'].value_counts()
    p = train_df['label'].value_counts(normalize=True)
    label = pd.concat([c,round(p*100, 2)], axis=1, keys=['counts', '%'])

    print(len(train_df))

    #AEDA
    aeda = AEDA(
        morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
    )        

    random.seed(42)

    #AEDA 진행
    data_aug = []
    for index, row in train_df.iterrows():
        result = []
        print(index)
        label_per = label.loc[row['label']]['%']
        if len(row['sentence']) > 200:
            continue
        if label_per < 10:
            result = aeda(row['sentence'])
            data_aug.append([result, row['subject_entity'], row['object_entity'], row['label'], row['source']])
            
            
        elif label_per < 1:
            result = aeda(row['sentence'], repeition = 2)
            for new in result:
                data_aug.append([new, row['subject_entity'], row['object_entity'], row['label'], row['source']])

    aug_df = pd.DataFrame(data_aug, columns=train_df.columns[1:])        
    
    print(len(aug_df))

    aug_df.to_csv("aug_df2.csv")

    #AEDA 후 불량 데이터 제거
    drop_idx = []
    for idx, row in aug_df.iterrows():
        subject_word = eval(row['subject_entity'])['word']
        object_word = eval(row['object_entity'])['word']
        if subject_word not in row['sentence'] or object_word not in row['sentence']:
            drop_idx.append(idx)
        
    print(len(drop_idx))

    aug_df = aug_df.drop(drop_idx)
    
    aug_df.to_csv("aug_df_dropped.csv", index=False)


if __name__ == "__main__":
    main()