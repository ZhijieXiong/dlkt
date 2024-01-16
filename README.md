# DLKT

[Document] | [Datasets] | [Models] | [中文版]

[Document]: DOC.md
[Datasets]: KT_DATASETS.md
[Models]: KT_MODELS.md
[中文版]: README_CN.md

## Introduction

## Quick-Start

1. Initialize project

   - Modify the environment configuration file `example/settings.json`

     ```python
     {
       "LIB_PATH": "F:\\code\\myProjects\\dlkt",  # Change to the project path, which is the directory where `lib` is located
       "FILE_MANAGER_ROOT": "F:\\code\\myProjects\\dlkt"  # Any path used to store data and models
     }
     ```

   - Run `set_up.py`

     ```shell
     python example/set_up.py
     ```

2. Place the original files of the dataset in the corresponding directory (Please refer to [Document](DOC.md) (Section 3) for details)

3. Data Preprocessing: Run ` example/reprocess.py`, for example

   ```shell
   python example/preprocess.py --dataset_name assist2009
   ```

4. Divide the dataset according to the specified experimental settings. For example, dividing the dataset according to the experimental setup of the AKT paper, i.e. 

   ```shell
   python example/prepare_dataset/akt_setting.py
   ```

   - For details on dataset partitioning, please refer to [Document](DOC.md)

5. Run the file under `example/train`, for example

   ```shell
   python example/train/dkt.py
   ```

   - Regarding the meaning of parameters, please refer to [Document](Doc.md)
