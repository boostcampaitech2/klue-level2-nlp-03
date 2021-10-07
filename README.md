# klue-level2-nlp-03
klue-level2-nlp-03 created by GitHub Classroom

## Train
```
$ python train.py --model_type=lstm --name=(wandb에 들어가는 이름)
```

## Inference
```
$ python inference.py --model_type=lstm
```

## 파일
bertmodel.py: Custom Trainer, Custom Model, Focal Loss가 구현되어있는 파일
- lstm+FocalLoss만 보실 분은 MyTrainer, Focal Loss, LSTMRobertaForSequenceClassification 코드만 확인 하셔도 무방
