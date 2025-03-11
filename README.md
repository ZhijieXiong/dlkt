[Document] | [DatasetInfo] | [PaperCollection] | [Experiment] | [RankingList]

[Document]: md_doc/DOC.md
[DatasetInfo]: https://zhijiexiong.github.io/sub-page/pyedmine/datasetInfo.html
[PaperCollection]: https://zhijiexiong.github.io/sub-page/pyedmine/paperCollection.html
[Experiment]: md_doc/Experiement.md
[RankingList]: https://zhijiexiong.github.io/sub-page/pyedmine/rankingList.html
NOTE

The experiments in the `Experiment` section are based on this code repository. Currently, I am in the process of refactoring the repository to eliminate redundant code and redesign the dataset division to create a more unified framework for **Knowledge Tracing (KT), Cognitive Diagnosis (CD), and Exercise Recommendation (ER) tasks**. As a result, the results in the `RankingList` may differ from those in the `Experiment` section. For the KT model, I have retained the hyperparameters used in the original experiments.

The new code repository is still under active development. Once completed, I will release all divided datasets, model training parameters, and model weights. Below is a flowchart illustrating the experimental design of the new repository.

If you are interested in accessing the new code repository, please feel free to email me, and I will grant you access.
# Introduction

A library of algorithms for reproducing knowledge tracing, cognitive diagnosis, and exercise recommendation models.

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

1. Divide the dataset according to the specified experimental settings: Run `example4knowledge_racing/prepare_dataset/akt_setting.py`. For example, 

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

1. Divide the dataset according to the specified experimental settings: Run `example4cognitive_diagnosis/prepare_dataset/akt_setting.py`. For example, 

   ```shell
   python ncd_setting.py
   ```

2. Train model: Run the file under `example4cognitive_diagnosis/train`. For example, train a NCD model

   ```shell
   python ncd.py
   ```

## Exercise Recommendation

1. Divide the dataset according to the specified experimental settings: Run `example4exercise_recommendation/prepare_dataset/kg4ex_setting.py`. For example, 

   ```
   python kg4ex_setting.py
   ```

2. Train or evaluate different model or method

   1. KG4EX
      - step 1, train a `DKT` model to get mlkc
      - step 2, train a `DKT_KG4EX` model to get pkc
      - step 3, run `example4exercise_recommendation/kg4ex/get_mlkc_pkc.py`
      - step 4, run `example4exercise_recommendation/kg4ex/get_efr.py`
      - step 5, run `example4exercise_recommendation/kg4ex/get_triples.py`
      - step 6, run `example4exercise_recommendation/train/kg4ex.py`

   2. EB-CF (Exercise-based collaborative filtering)
      - step1, change `example4exercise_recommendation/eb_cf/load_data` to get users' history data
      - step2, run `example4exercise_recommendation/eb_cf/get_que_sim_mat.py` to get questions' similarity matrix
      - step3, run `example4exercise_recommendation/eb_cf/evaluate.py`
   3. SB-CF (Student-based collaborative filtering)
      - Similar to EB-CF, run the code in `example4exercise_recommendation/sb_cf`

# Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/ZhijieXiong/dlkt/issuesWe) 

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

# Contributors

- https://github.com/ZhijieXiong
- https://github.com/kingofpop625
- https://github.com/shshen-closer
