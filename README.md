# DLKT

[Doc] | [Datasets] | [Models] | [中文版]

[Doc]: DOC.md
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

2. Place the original files of the dataset in the corresponding directory
