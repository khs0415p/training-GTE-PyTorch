# General Text Embeddings

논문에 대한 설명은 [블로그](https://velog.io/@khs0415p/Paper-GTE)를 참고하시기 바랍니다.

## Abstract

**Improved contrastive loss**과 다양한 도메인의 데이터를 활용하여 학습만 모델이며 BERT와 같은 bidirectional 모델을 베이스로 사용하지만 현재 기존의 모델들 뿐만 아니라 LLM들 사이에서도 좋은 성능을 보여주고 있다.

## Dataset

dataset은 _(q, d)_ positive pair로 구성하여 학습을 수행한다.

## Tree

```
.
├── config                          # folder for config files
│   └── config.yaml
│
├── data                            # data file
│   └── data.pk
│
├── main                            # file for gte runing
│   └── run_gte.py
│
├── models                          # gte modeling file
│   ├── __init__.py
│   ├── configuration_gte.py
│   └── modeling_gte.py
│
├── results                         # folder for checkpoint
│   └── checkpoint
│   
├── trainer                         
│   ├── __init__.py
│   ├── base.py                     # base trainer for inheritance
│   └── trainer.py                  # gte trainer
│
├── utils
│   ├── __init__.py
│   ├── data_utils.py               # dataset
│   ├── file_utils.py
│   └── train_utils.py
│
└── requirements.txt
```

## Start

```
python main/run_gte.py --config config/config.yaml
```
