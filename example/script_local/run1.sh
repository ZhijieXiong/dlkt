#!/usr/bin/env bash

python F:/code/myProjects/dlkt/example/preprocess.py --dataset_name "junyi2015"
python F:/code/myProjects/dlkt/example/prepare_dataset/lbkt_setting.py --dataset_name "junyi2015"
bash F:/code/myProjects/dlkt/example/script_local/lbkt/lbkt_setting_junyi2015.sh
