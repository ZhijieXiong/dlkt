[TOC]

# DLKT

[Document] | [Datasets] | [Models] | [中文版]

[Document]: md_doc/DOC.md
[Datasets]: md_doc/KT_DATASETS.md
[Models]: md_doc/MODELS.md
[中文版]: md_doc/README_CN.md

# Introduction

This algorithm library was developed inspired by `PYKT` (the code repository corresponding to the paper `"PYKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models", NeurIPS 2022`). `PYKT` provides great convenience to researchers in the field of `Knowledge Tracing` and has made outstanding contributions to the development of `Knowledge Tracing`. However, we believe that `PYKT` also has some shortcomings, so we opened this algorithm library, named `DLKT`, to achieve complementary effects with `PYKT`.

- The Comparison with `PYKT`

|                                                              | DLKT  | PYKT  |
| ------------------------------------------------------------ | ----- | ----- |
| Arbitrary Experimental Settings (Data Processing, Dataset  Partitioning, ...) | True  | False |
| Mainstream Knowledge Tracing Model (DKT, DKVMN, AKT, LPKT, ...) | True  | True  |
| Knowledge Tracing Model  Training Based on Concepts          | True  | True  |
| Knowledge Tracing Model Training Based on Questions          | True  | True  |
| Knowledge Tracing Model  Evaluating Based on Concepts        | True  | True  |
| Knowledge Tracing Model  Evaluating Based on Questions       | True  | True  |
| Fine-grained Metrics (Cold Start, Long Tail, ...)            | True  | False |
| Knowledge Tracing Model Training Similar to Sequential Recommendation System | True  | False |
| Cognitive Diagnosis Task                                     | True  | False |
|                                                              |       |       |
| Wandb                                                        | False | True  |

# Experiement Results

## PYKT settings (Knowledge Tracing)

- All scripts for this experiment are in the `example\script_local\pykt_repo` directory.

- To view the complete experimental record, please click [here](https://docs.qq.com/sheet/DREdTc3lVQWJkdFJw?tab=BB08J2).

- All reproduction results are based on adjusting parameters at 1 fold (**Parameter adjustment is performed under the `example/prepare_datset/our_setting` experimental setting. We use the parameters of `our_setting` directly under `pykt_question_setting`, so the reproduction results are somewhat different from the results reported in the paper.**), and then taking the average of 5 folds (in order to reduce randomness, the random seeds of all experiments are fixed to 0). The values in parentheses are the results reported in the paper. The metric in the table is `AUC`.

- The results reported in the paper come from `PYKT`, `SimpleKT` `AT-DKT` and `QIKT`. Please see [the corresponding paper](md_doc/MODELS.md).

- Reproduction results on `multi concept` datasets. Please note: 

  1. We did not first extend the exercise sequence into a knowledge concept sequence like `PYKT` (`PYKT` paper picture 2), then train the model on the knowledge concept sequence, and finally test the model on the question (`PYKT` paper Section 3.3). We reproduce by training and testing the model directly on the question sequence, that is, for multi-concept questions, we use `mean pooling` for multiple concept embeddings.
  2. This difference is not only reflected in the training and testing of the model, but also in the data preprocessing. `PYKT` first extends the sequence and then cuts the sequence, fixing the length of each sequence to 200. We cut the sequence directly, with a fixed sequence length of 200.
  3. We currently only implement average pooling of concepts on `DKT, AKT, SimpleKT, DIMKT, and QIKT` models. In addition, because the original code of `ATKT` has data leakage problems, we use the `atktfix` provided by `PYKT`.

  |          |   Assist2009   |     AL2005     |     BD2006     |    xes3g5m     |
  | :------: | :------------: | :------------: | :------------: | :------------: |
  |   DKT    | 0.756(0.7541)  | 0.8162(0.8149) | 0.7748(0.8015) | 0.7849(0.7852) |
  |   AKT    | 0.7911(0.7853) | 0.8169(0.8306) | 0.8162(0.8208) | 0.8231(0.8207) |
  | SimpleKT |                |                |                |                |
  |   QIKT   | 0.7907(0.7878) |                |                |                |
  |   qDKT   |     0.7762     |     0.8363     |     0.8144     | 0.8261(0.8225) |

- Reproduction results on `single concept` datasets. Please note: 

  1. For datasets with a small number of questions, our DKT and ATKT also provide results with questions as items.
  2. For the `statics2011` and `edi2020-task34` dataset, our data preprocessing is different from `PYKT`

  |           |  Statics2011   |     NIPS34     |
  | :-------: | :------------: | :------------: |
  |    DKT    |     0.7142     | 0.762(0.7681)  |
  |  DKT_que  | 0.8161(0.8222) | 0.7935(0.7995) |
  |   DKVMN   |     0.7066     | 0.7512(0.7673) |
  | DKVMN_que |                |                |
  |   ATKT    |                |                |
  | ATKT_que  |                |                |
  |    AKT    | 0.8244(0.8309) | 0.7943(0.8033) |
  | SimpleKT  |                |                |
  |  AT-DKT   |                |                |
  |   QIKT    |                | 0.7993(0.8044) |
  |   qDKT    |     0.8236     |     0.7968     |

## Our Setting (Knowledge Tracing)

- To view the complete experimental record, please click [here](https://docs.qq.com/sheet/DRG50d3B5VlBJdGpK?tab=BB08J2).

- The experimental results of the `AUC` are as follows:

  |          | Assist2009 | Algebra2005 | Bridge2Algebra2006 | Ednet-kt1 | Xes3g5m |
  | :------: | :--------: | :---------: | :----------------: | :-------: | :-----: |
  |   DKT    |   0.7484   |   0.8163    |       0.7768       |  0.6585   | 0.7842  |
  |   AKT    |   0.786    |    0.823    |       0.8163       |  0.7327   | 0.8215  |
  |   LPKT   |            |             |                    |  0.7381   |         |
  |  DIMKT   |   0.7647   |             |                    |  0.7109   |         |
  | SimpleKT |   0.7848   |   0.8431    |                    |           |         |
  |   QIKT   |   0.7838   |             |                    |           |         |
  |   qDKT   |   0.7657   |    0.835    |       0.8166       |   0.729   | 0.8254  |

  |          | Assist2012 | Assist2015 | Assist2017 | Statics2011 | Slepemapy | Edi2020-task34 |
  | :------: | :--------: | :--------: | :--------: | :---------: | :-------: | :------------: |
  |   DKT    |   0.7333   |            |   0.7296   |   0.7117    |  0.6844   |     0.7598     |
  | DKT_que  |            |            |   0.795    |   0.8148    |           |     0.7912     |
  |  DKVMN   |   0.7197   |            |   0.6944   |   0.7042    |  0.6732   |     0.7479     |
  |   ATKT   |            |            |   0.7206   |   0.6943    |  0.6692   |     0.7569     |
  | ATKT_que |            |            |   0.7394   |   0.7961    |           |     0.7834     |
  |   AKT    |   0.7905   |            |   0.7732   |   0.8201    |  0.7285   |     0.793      |
  |   LPKT   |   0.7889   |            |   0.811    |             |  0.7373   |                |
  |  DIMKT   |   0.7854   |            |   0.801    |   0.8202    |  0.7283   |     0.7943     |
  | SimpleKT |            |            |   0.7741   |    0.822    |  0.7316   |     0.7931     |
  |  AT-DKT  |            |            |            |             |           |                |
  |   QIKT   |            |            |   0.7879   |   0.8266    |  0.7236   |     0.7972     |
  |   qDKT   |   0.7855   |            |   0.7921   |   0.8199    |   0.724   |     0.7952     |


## NCD Setting (Cognitive Diagnosis)

- paper: `"Neural Cognitive Diagnosis for Intelligent Education Systems"`	

| Assist2009 | AUC    | ACC    | RMSE   |
| ---------- | ------ | ------ | ------ |
| paper      | 0.749  | 0.719  | 0.439  |
| repo       | 0.7551 | 0.7236 | 0.4328 |



# Referrence

[papers](md_doc/MODELS.md)

[datasets](md_doc/KT_DATASETS.md)

1. https://github.com/pykt-team/pykt-toolkit
2. https://github.com/bigdata-ustc/EduKTM
3. https://github.com/bigdata-ustc/EduData
4. https://github.com/arghosh/AKT
5. https://github.com/xiaopengguo/ATKT
6. https://github.com/shshen-closer/DIMKT
7. https://github.com/UpstageAI/cl4kt
8. https://github.com/yxonic/DTransformer
9. https://github.com/garyzhao/ME-ADA
10. https://github.com/RuihongQiu/DuoRec
11. https://github.com/salesforce/ICLRec
12. https://github.com/YChen1993/CoSeRec
13. https://github.com/rlqja1107/MELT
14. https://github.com/QinHsiu/MCLRec

# Quick-Start

## Prepare

1. Initialize project

   - Create file `settings.json` in the root directory.

   - Modify the environment configuration file `settings.json`

     ```python
     {
       "LIB_PATH": ".../dlkt-main",  # Change to the project root path
       "FILE_MANAGER_ROOT": "any_dir"  # Any path used to store data and models
     }
     ```

   - Run `set_up.py`

     ```shell
     python set_up.py
     ```

2. Place the original files of the dataset in the corresponding directory (Please refer to [Document (Section 1.3)](md_doc/DOC.md) for details)

3. Data Preprocessing: Run ` example/preprocess.py`, for example

   ```shell
   python preprocess.py --dataset_name assist2009
   ```

## Knowledge Tracing

1. Divide the dataset according to the specified experimental settings: Run `example/prepare_dataset/akt_setting.py`. For example, dividing the dataset according to the experimental setup of the AKT paper, i.e. 

   ```shell
   python akt_setting.py
   ```

   - For details on dataset partitioning, please refer to [Document (Section 1.6)](md_doc/DOC.md)

2. Train model: Run the file under `example/train`. For example, train a DKT model

   ```shell
   python dkt.py
   ```

   - Regarding the meaning of parameters, please refer to [Document (Section 2)](Doc.md)

## Cognitive Diagnosis

1. Divide the dataset according to the specified experimental settings: Run `example4cognitive_diagnosis/prepare_dataset/akt_setting.py`. For example, dividing the dataset according to the experimental setup of the AKT paper, i.e. 

   ```shell
   python ncd_setting.py
   ```

2. Train model: Run the file under `example4cognitive_diagnosis/train`. For example, train a NCD model

   ```shell
   python ncd.py
   ```


# Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/ZhijieXiong/dlkt/issuesWe) 

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

# Contributors

- https://github.com/ZhijieXiong
- https://github.com/kingofpop625
- https://github.com/shshen-closer
