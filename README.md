# DLKT

[Document] | [Datasets] | [Models] | [中文版]

[Document]: DOC.md
[Datasets]: KT_DATASETS.md
[Models]: MODELS.md
[中文版]: README_CN.md

## Introduction

## Quick-Start

1. Initialize project

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
