<h1 align="center">Entity Relation Extraction in sentences 👋</h1>

<!-- <p align="center">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/boostcampaitech2/klue-level2-nlp-03?style=social">
  <img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/boostcampaitech2/klue-level2-nlp-03?style=plastic">
  <img alt="Conda" src="https://img.shields.io/conda/pn/boostcampaitech2/klue-level2-nlp-03">
</p>   -->

## Overview Description

Relation extraction (RE) identifies semantic relations between entity pairs in a text. The relation is defined between an entity pair consisting of subject entity ($$e_{\text{subj}}$$) and object entity ($e_{\text{obj}}$ ). For example, in a sentence 'Kierkegaard was born to an affluent family in Copenhagen’, the subject entity is `Kierkegaard` and the object entity is `Copenhagen`. The goal is then to pick an appropriate relationship between these two entities: $place\_of\_birth$. In order to evaluate whether a model correctly understands the relationships between entities, we include KLUE-RE in our benchmark. Since there is no large-scale RE benchmark publicly available in Korean, we collect and annotate our own dataset.


We formulate RE as a single sentence classification task. A model picks one of predefined relation types describing the relation between two entities within a given sentence. In other words, an RE model predicts an appropriate relation $r$ of entity pair $(e_{\text{subj}},\ e_{\text{obj}})$ in a sentence $s$, where $e_{\text{subj}}$ is the subject entity and $e_{\text{obj}}$ is the object entity. We refer to $(e_{\text{subj}},\ r,\ e_{\text{obj}})$ as a relation triplet. The entities are marked as corresponding spans in each sentence $s$. There are 30 relation classes that consist of 18 person-related relations, 11 organization-related relations, and $\textit{no_relation}$. We evaluate a model using micro-F1 score, computed after excluding $\textit{no_relation}$, and area under the precision-recall curve (AUPRC) including all 30 classes.

## Evaluation Methods
The evaluation metrics for KLUE-RE are 1) micro F1 score on relation existing cases, and 2) area under the precision-recall curve (AUPRC) on all classes.


Micro F1 score is a geometric mean of micro-precision and micro-recall. It measures the F1-score of the aggregated contributions of all classes. It gives each sample the same importance, thus naturally weighting more on the majority class. We remove the dominant class $(no\_relation)$ for this metric to not incentivize the model predicting negative class very well.


AUPRC is an averaged area under the precision-recall curves whose x-axis is recall and y-axis is the precision of all relation classes. It is a useful metric for this imbalanced data setting while rare positive examples are important.

## Code Contributors

<p>
<a href="https://github.com/iamtrueline" target="_blank">
  <img x="5" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/79238023?v=4"/>
</a>
<a href="https://github.com/promisemee" target="_blank">
  <img x="74" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/31719240?v=4"/>
</a>
<a href="https://github.com/kimminji2018" target="_blank">
  <img x="143" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/74283190?v=4"/>
</a>
<a href="https://github.com/Ihyun" target="_blank">
  <img x="212" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/32431157?v=4"/>
</a>
<a href="https://github.com/sw6820" target="_blank">
  <img x="281" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/52646313?v=4"/>
</a>
<a href="https://github.com/NayoungLee-de" target="_blank">
  <img x="350" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/69383548?v=4"/>
</a>

</p>

## Environments 

### OS
 - UBUNTU 18.04

### Requirements
- python==3.8
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.10.0


### Hardware
The following specs were used to create the original solution.
- GPU(CUDA) : v100 

## Reproducing Submission
To reproduct my submission without retraining, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Prepare Datasets](#Prepare-Datasets)
4. [Download Baseline Codes](#Download-Baseline-Codes)
5. [Train models](#Train-models-(GPU-needed))
6. [Inference & make submission](#Inference-&-make-submission)
7. [Ensemble](#Ensemble)
8. [Wandb graphs](#Wandb-graphs)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
$ pip install -r requirements.txt
```

## Dataset Preparation
All CSV files are already in data directory.


### Prepare Datasets
After downloading and converting datasets and baseline codes, the data directory is structured as:
```
├── code
│   ├── __pycache__
│   │    └── load_data.cpython-38.pyc
│   ├── wandb_imgaes
│   │    ├── eval.png 
│   │    ├── eval2.png
│   │    ├── train.png
│   │    ├── train2.png
│   │    ├── system.png
│   │    ├── system2.png
│   │    └── system3.png
│   ├── best_model
│   ├── ensemble_csv
│   ├── dict_label_to_num.pkl
│   ├── dict_num_to_label.pkl
│   ├── inference.py
│   ├── load_data.py
│   ├── bertmodel.py
│   ├── logs
│   ├── prediction
│   │    └── sample_submission.csv
│   ├── requirements.txt
│   ├── results
│   └── train.py
└── dataset
    ├── test
    │    └── test_data.csv    
    └── train
         └── train.csv
```
#### Download Baseline code
To download baseline codes, run following command. The baseline codes will be located in *opt/ml/code*
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000075/data/code.tar.gz
```

#### Download Dataset
To download dataset, run following command. The dataset will be located in *opt/ml/dataset*
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000075/data/dataset.tar.gz
``` 
### Train models (GPU needed)
To train models, run following commands.
```
$ python train.py 
```
The expected training times are:

Model | GPUs | Batch Size | Training Epochs | Training Time
------------  | ------------- | ------------- | ------------- | -------------
KoELECTRA | v100 | 16 | 4 | 1h 51m 29s
XLM-RoBERTa-large | v100 | 27 | 4 | 2h 26m 52s
LSTM-RoBERTa-large | v100 | 32 | 5 |  2h 25m 14s
RoBERTa-large | v100 | 32 | 5 | 2h 5m 23s


### Inference & make submission
```
$ python inference.py
```

### Ensemble
```
$python ensemble.py --path='./ensemble_csv'
```

### Wandb graphs
- eval graphs
<p>
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/eval.png">
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/eval2.png">
</p>    

- train graphs
<p>
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/train.png">
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/train2.png">
</p>    

- system graphs
<p>
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system.png">
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system2.png">
    <img src="https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system3.png">
</p>
