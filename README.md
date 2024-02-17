# DLKT

[Document] | [Datasets] | [Models] | [中文版]

[Document]: DOC.md
[Datasets]: KT_DATASETS.md
[Models]: MODELS.md
[中文版]: README_CN.md

# Introduction



# Referrence

## Projects

[papers](MODELS.md)

[datasets](KT_DATASETS.md)

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

1. Initialize project

   - Create file `settings.json` in the `example` directory.

   - Modify the environment configuration file `example/settings.json`

     ```python
     {
       "LIB_PATH": ".../dlkt-main",  # Change to the project root path
       "FILE_MANAGER_ROOT": "any_dir"  # Any path used to store data and models
     }
     ```

   - Run `example/set_up.py`

     ```shell
     python set_up.py
     ```

2. Place the original files of the dataset in the corresponding directory (Please refer to [Document (Section 1.3)](DOC.md) for details)

3. Data Preprocessing: Run ` example/preprocess.py`, for example

   ```shell
   python preprocess.py --dataset_name assist2009
   ```

4. Divide the dataset according to the specified experimental settings: Run `example/prepare_dataset/akt_setting.py`. For example, dividing the dataset according to the experimental setup of the AKT paper, i.e. 

   ```shell
   python akt_setting.py
   ```

   - For details on dataset partitioning, please refer to [Document (Section 1.6)](DOC.md)

5. Train model: Run the file under `example/train`. For example, train a DKT model

   ```shell
   python dkt.py
   ```

   - Regarding the meaning of parameters, please refer to [Document (Section 2)](Doc.md)

# Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/ZhijieXiong/dlkt/issuesWe) 

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

# Contributors

- https://github.com/ZhijieXiong
- https://github.com/kingofpop625
