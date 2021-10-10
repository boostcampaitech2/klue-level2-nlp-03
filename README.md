<h1 align="center">Entity Relation Extraction in sentences 👋</h1>

## Environments 

### Requirements
- python==3.8
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.10.0


### Hardware
The following specs were used to create the original solution.
- GPU(CUDA) : V100 

## Reproducing Submission
To reproduct my submission without retraining, do the following steps:
<!-- 1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Make RGBY Images](#make-rgby-images) for official.
4. [Download Pretrained models](#pretrained-models)
5. [Inference](#inference)
6. [Make Submission](#make-submission) -->

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
pip install -r requirements.txt
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
#### eval graphs
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/eval.png)"></a>
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/eval2.png)"></a>
#### train graphs
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/train.png)"></a>
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/train2.png)"></a>
#### system graphs
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system.png)"></a>
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system2.png)"></a>
<img src="[images/200x200.png](https://github.com/boostcampaitech2/klue-level2-nlp-03/blob/Main/wandb_imgaes/system3.png)"></a>

## Code Contributors

This project exists thanks to all the people who contribute. 
<img src="https://sw6820.github.io/ai_boostcamp_lv3_3/" /></a>
