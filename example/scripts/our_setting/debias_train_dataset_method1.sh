#!/usr/bin/env bash

setting_name="our_setting_new"
dataset_names=("assist2009" "assist2012" "assist2017" "ednet-kt1" "statics2011")
# "slepemapy" "junyi2015" "edi2020-task34" "edi2020-task1"
folds=(0 1 2 3 4)

for dataset_name in "${dataset_names[@]}"
do
  for fold in "${folds[@]}"
  do
    python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_debias_method1.py \
      --max_num2delete 20 --additional_delete 1 \
      --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/${dataset_name}_train_fold_${fold}.txt"
  done
done