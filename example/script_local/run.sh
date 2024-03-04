#!/usr/bin/env bash

python F:/code/myProjects/dlkt/example/prepare_dataset/akt_setting.py --dataset_name "assist2009"
python F:/code/myProjects/dlkt/example/prepare_dataset/akt_setting.py --dataset_name "assist2017"
python F:/code/myProjects/dlkt/example/prepare_dataset/akt_setting.py --dataset_name "statics2011"
python F:/code/myProjects/dlkt/example/prepare_dataset/akt_table6_setting.py --dataset_name "assist2009"
python F:/code/myProjects/dlkt/example/prepare_dataset/dimkt_setting.py --dataset_name "assist2012"
python F:/code/myProjects/dlkt/example/prepare_dataset/dimkt_setting.py --dataset_name "edi2020-task1"
python F:/code/myProjects/dlkt/example/prepare_dataset/lpkt_setting.py --dataset_name "assist2012"
python F:/code/myProjects/dlkt/example/prepare_dataset/lpkt_setting.py --dataset_name "assist2017"
