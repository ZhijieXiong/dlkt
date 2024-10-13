[TOC]

# DLKT

[Document] | [Datasets] | [Models] | [中文版]

[Document]: md_doc/DOC.md
[Datasets]: md_doc/KT_DATASETS.md
[Models]: md_doc/MODELS.md
[中文版]: md_doc/README_CN.md

# Introduction

This algorithm library was developed inspired by `pyKT` (the code repository corresponding to the paper [`"pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models", NeurIPS 2022`](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract-Datasets_and_Benchmarks.html)). `pyKT` provides great convenience to researchers in the field of `Knowledge Tracing` and has made outstanding contributions to the development of `Knowledge Tracing`. However, we believe that `pyKT` also has some shortcomings, so we opened this algorithm library, named `DLKT`, to achieve complementary effects with `pyKT`.

- The Comparison with `pyKT`

|                                     Content                                     | DLKT  | pyKT  |
|:-------------------------------------------------------------------------------:| ----- | ----- |
|  Arbitrary Experimental Settings (Data Processing, Dataset  Partitioning, ...)  | True  | False |
|         Mainstream Knowledge Tracing Model (DKT, DKVMN, AKT, LPKT, ...)         | True  | True  |
|               Knowledge Tracing Model  Training Based on Concepts               | True  | True  |
|               Knowledge Tracing Model Training Based on Questions               | True  | True  |
|              Knowledge Tracing Model  Evaluating Based on Concepts              | True  | True  |
|             Knowledge Tracing Model  Evaluating Based on Questions              | True  | True  |
| Fine-grained Metrics (Cold Start, Long Tail, Biased and Unbiased Evaluation...) | True  | False |
|  Knowledge Tracing Model Training Similar to Sequential Recommendation System   | True  | False |
|                            Cognitive Diagnosis Task                             | True  | False |
|                                                                                 |       |       |
|                                      Wandb                                      | False | True  |

- Data process pipline

<div align=center><img src="./resources/kt_data_pipline.png" /></div>

# Experiement Results

## Our setting (Knowledge Tracing)

### Overall metric (AUC)

|           | Assist2009 | Assist2012 | Statics2011 | Ednet-KT1 | Xes3g5m | Slemapy-Anatomy |
| :-------: | :--------: | :--------: | :---------: | :-------: | :-----: | :-------------: |
|    DKT    |   0.7481   |   0.7337   |   0.6612    |  0.7113   | 0.7838  |     0.6838      |
|   DKVMN   |   0.7456   |   0.7217   |    0.668    |  0.7046   | 0.7748  |     0.6745      |
|   SAKT    |   0.7328   |   0.721    |   0.6642    |  0.6776   | 0.7791  |      0.676      |
|   LPKT    |   0.7682   |   0.7884   |   0.7394    |  0.8216   | 0.8257  |     0.7365      |
|   DIMKT   |   0.7647   |   0.7845   |   0.7154    |  0.8198   | 0.8262  |     0.7285      |
| SimpleKT  |   0.7853   |   0.7818   |   0.7342    |  0.8214   | 0.8198  |     0.7316      |
|   QIKT    |   0.7843   |   0.7753   |   0.7329    |  0.8268   | 0.8232  |     0.7234      |
| SparseKT  |   0.7782   |   0.7727   |   0.7302    |  0.8162   | 0.8153  |     0.7258      |
|   MIKT    |   0.7886   |   0.7902   |   0.7436    |  0.8213   | 0.8179  |     0.7369      |
|    AKT    |   0.7854   |   0.7904   |   0.7345    |  0.8193   | 0.8225  |     0.7288      |
|   qDKT    |   0.7684   |   0.7861   |   0.7354    |  0.8191   | 0.8252  |     0.7247      |
| AKT-CORE  |   0.7512   |   0.7619   |   0.7076    |  0.7811   | 0.8037  |     0.7133      |
| qDKT-CORE |   0.7365   |   0.7527   |   0.6544    |  0.7608   |  0.78   |     0.7008      |

|          | Assist2017 | Junyi2015 |    Edi2020-task1    |   Edi2020-task34    |
| :------: | :--------: | :-------: | :-----------------: | :-----------------: |
|   qDKT   |   0.7919   |  0.7806   |       0.8141        |       0.7947        |
|   AKT    |   0.772    |  0.7791   |       0.8129        |        0.793        |
|   LPKT   |   0.812    |           |       0.8179        |       0.7968        |
|  DIMKT   |   0.8002   |  0.7836   |       0.8138        |       0.7936        |
| SimpleKT |   0.7746   |  0.7793   |       0.8135        |       0.7937        |
|   QIKT   |   0.7874   |  0.7812   |         OOM         |       0.7972        |
|   LBKT   |   0.8335   |  0.7829   | Lack of information | Lack of information |

`LBKT` in `Assist2009`: 0.7767

`LBKT` in `Assist2012`: 0.7914

### CORE metric (AUC)

|           | Assist2009 | Assist2012 | Statics2011 | Statics | Xes3g5m | Slemapy-Anatomy |
| :-------: | :--------: | :--------: | :---------: | :-----: | :-----: | :-------------: |
|    DKT    |   0.6931   |   0.6716   |   0.5857    | 0.6447  | 0.7031  |     0.6681      |
|   DKVMN   |   0.6859   |   0.6615   |   0.5817    | 0.6468  | 0.6979  |     0.6622      |
|   SAKT    |   0.6755   |   0.6582   |   0.5806    | 0.6283  | 0.7003  |     0.6651      |
|   LPKT    |   0.6559   |   0.6684   |   0.5712    | 0.6061  | 0.7097  |     0.6789      |
|   DIMKT   |   0.6821   |   0.664    |   0.5671    | 0.6131  | 0.7102  |     0.6679      |
| SimpleKT  |   0.6903   |   0.6607   |   0.5722    | 0.6155  | 0.7002  |     0.6712      |
|   QIKT    |   0.6776   |   0.6469   |   0.5652    | 0.6262  | 0.7076  |     0.6634      |
| SparseKT  |   0.6754   |   0.6438   |   0.5591    | 0.6025  | 0.6914  |     0.6667      |
|   MIKT    |   0.6874   |   0.6673   |   0.5809    | 0.6161  | 0.6895  |     0.6791      |
|    AKT    |   0.6955   |   0.6789   |   0.5769    | 0.6173  | 0.7127  |     0.6776      |
|   qDKT    |   0.6826   |   0.666    |   0.5708    | 0.6146  | 0.7078  |      0.665      |
| AKT-CORE  |   0.6966   |   0.6965   |   0.5858    | 0.6319  | 0.7315  |     0.6902      |
| qDKT-CORE |   0.6657   |   0.6502   |   0.5709    | 0.5864  | 0.6806  |     0.6396      |

## Our DG setting (Knowledge Tracing)

- DG (Doamin Generalization) setting: Divide the data into training data and testing data according to the student attributes (school, country, etc.). For example, the student data of schools A, B, and C are used as the training set, and the student data of school D is used as the testing set.
- The `Assist2009` and `Assist2012` datasets have student school information, so domain generalization experiments can be conducted based on schools. The experimental settings are as follows:
  1. Merge schools with small numbers of students into one school, and do not use extreme schools (with an average sequence length of less than 20) as testing set
  2. After the merger is completed, first randomly divide 80% of the schools into training sets based on the school as a unit, and the number of samples in the training set is required to account for 70% to 85% of the total number of samples
  3. After dividing the training set and the test set, 20% of the training set is divided as the validation set based on students
  4. Obtain 10 different divisions through random division, and use qDKT to measure the gap between I.I.D. and O.O.D, that is, the performance gap between the model on the validation set and the testing set
  5. The division result with the largest gap is selected for the experiment. In order to reduce randomness, the reported result is the average of the results of 5 random seeds
  6. The method for stopping model training is still early stop, and the model with the highest performance in the validation set is selected
  7. Because the validation set is I.I.D., the parameters of each model are the same as those in the previous experiment, and no parameters are adjusted
- The `Sleepmapy-Anatomy` data set has student country information, so domain generalization experiments can be performed based on countries. Since the student data of one country in the data set accounts for 80%, the student data of this country is directly used as the training set, and the remaining data is used as the testing set.
- Report results: The results outside the brackets in the table are the validation set results, and the results inside the brackets are the testing set result.

### AUC Metric

|       | Assist2009      | Assist2012      | Slepemapy-Anatomy |
| ----- | --------------- | --------------- | ----------------- |
| qDKT  | 0.7482 (0.7327) | 0.7748 (0.7523) | 0.7258 (0.7096)   |
| AKT   | 0.7558 (0.7321) | 0.7766 (0.7506) | 0.7303 (0.7129)   |
| LPKT  | 0.7525 (0.7416) | 0.7787 (0.7577) | 0.7423 (0.7238)   |
| DIMKT | 0.7247 (0.7386) | 0.7724 (0.7449) | 0.7303 (0.7139)   |
| LBKT  | 0.7603 (0.7482) | 0.779 (0.7574)  |                   |

## pyKT setting (Knowledge Tracing)

- To view the complete experimental record, please click [here](https://docs.qq.com/sheet/DREdTc3lVQWJkdFJw?tab=BB08J2).

- All reproduction results are based on adjusting parameters at 1 fold (**Parameter adjustment is performed under the `example/prepare_datset/our_setting` experimental setting. We use the parameters of `our_setting` directly under `pykt_question_setting`, so the reproduction results are somewhat different from the results reported in the paper.**), and then taking the average of 5 folds (in order to reduce randomness, the random seeds of all experiments are fixed to 0). The values in parentheses are the results reported in the paper. The metric in the table is `AUC`.

- The results reported in the paper come from `pyKT`, `SimpleKT` `AT-DKT` and `QIKT`. Please see [the corresponding paper](md_doc/MODELS.md).

- Reproduction results on `multi concept` datasets. Please note: 

  1. We did not first extend the exercise sequence into a knowledge concept sequence like `pyKT` (`pyKT` paper picture 2), then train the model on the knowledge concept sequence, and finally test the model on the question (`pyKT` paper Section 3.3). We reproduce by training and testing the model directly on the question sequence, that is, for multi-concept questions, we use `mean pooling` for multiple concept embeddings.
  2. This difference is not only reflected in the training and testing of the model, but also in the data preprocessing. `pyKT` first extends the sequence and then cuts the sequence, fixing the length of each sequence to 200. We cut the sequence directly, with a fixed sequence length of 200.
  3. We currently only implement average pooling of concepts on `DKT, AKT, SimpleKT, DIMKT, and QIKT` models. In addition, because the original code of `ATKT` has data leakage problems, we use the `atktfix` provided by `pyKT`.

  |          |   Assist2009   |     AL2005     |     BD2006     |    xes3g5m     |
  | :------: | :------------: | :------------: | :------------: | :------------: |
  |   DKT    | 0.756(0.7541)  | 0.8162(0.8149) | 0.7748(0.8015) | 0.7849(0.7852) |
  |   AKT    | 0.7911(0.7853) | 0.8169(0.8306) | 0.8162(0.8208) | 0.8231(0.8207) |
  | SimpleKT | 0.7906(0.7744) | 0.8426(0.8254) | 0.8144(0.816)  | 0.821(0.8163)  |
  |   QIKT   | 0.7907(0.7878) |      OOM       |      OOM       |      todo      |
  |   qDKT   |     0.7762     |     0.8363     |     0.8144     | 0.8261(0.8225) |

- Reproduction results on `single concept` datasets. Please note: 

  1. For datasets with a small number of questions, our DKT and ATKT also provide results with questions as items.
  2. For the `statics2011` and `edi2020-task34` dataset, our data preprocessing is different from `pyKT`

  |           |  Statics2011   |     NIPS34     |
  | :-------: | :------------: | :------------: |
  |    DKT    |     0.7142     | 0.762(0.7681)  |
  |  DKT_que  | 0.8161(0.8222) | 0.7935(0.7995) |
  |   DKVMN   |     0.7066     | 0.7512(0.7673) |
  | DKVMN_que | 0.8078(0.8093) |     0.7901     |
  |   SAINT   | 0.7273(0.7599) | 0.7846(0.7873) |
  |   ATKT    |     0.696      | 0.7603(0.7665) |
  | ATKT_que  | 0.8018(0.8055) |     0.7844     |
  |    AKT    | 0.8244(0.8309) | 0.7943(0.8033) |
  | SimpleKT  | 0.8258(0.8199) | 0.7955(0.8035) |
  |  AT-DKT   |      todo      |      todo      |
  |   QIKT    |     0.8303     | 0.7993(0.8044) |
  |   qDKT    |     0.8236     |     0.7968     |

## Other Setting (Knowledge Tracing)

- To view the complete experimental record, please click [here](https://docs.qq.com/sheet/DREtXSUtqTkZrTVVY?tab=BB08J2)

## NCD Setting (Cognitive Diagnosis)

- paper: `"Neural Cognitive Diagnosis for Intelligent Education Systems"`	

| Assist2009 | AUC    | ACC    | RMSE   |
| ---------- | ------ | ------ | ------ |
| paper      | 0.749  | 0.719  | 0.439  |
| repro      | 0.7551 | 0.7236 | 0.4328 |

# Referrence

- [paper](md_doc/MODELS.md)

- [dataset](md_doc/KT_DATASETS.md)

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
