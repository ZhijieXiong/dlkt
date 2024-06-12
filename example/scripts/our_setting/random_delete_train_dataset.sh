#!/usr/bin/env bash

setting_name="our_setting_new"
# "slepemapy" "junyi2015" "edi2020-task34" "edi2020-task1"
folds=(0 1 2 3 4)

#for fold in "${folds[@]}"
#do
#  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_random_delete.py \
#    --max_sample_delete 100 \
#    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/assist2009_train_fold_${fold}.txt"
#done

for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_random_delete.py \
    --max_sample_delete 116984 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/assist2012_train_fold_${fold}.txt"
done

for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_random_delete.py \
    --max_sample_delete 28542 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/assist2017_train_fold_${fold}.txt"
done

for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_random_delete.py \
    --max_sample_delete 13503 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/ednet-kt1_train_fold_${fold}.txt"
done

#for fold in "${folds[@]}"
#do
#  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_random_delete.py \
#    --max_sample_delete 100 \
#    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/statics2011_train_fold_${fold}.txt"
#done