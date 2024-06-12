#!/usr/bin/env bash

setting_name="our_setting_new"
folds=(0 1 2 3 4)

for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_debias_method2.py \
    --max_seq_len2delete 200 --acc_th 0.18 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/assist2012_train_fold_${fold}.txt"
done


for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_debias_method2.py \
    --max_seq_len2delete 200 --acc_th 0.32 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/assist2017_train_fold_${fold}.txt"
done


for fold in "${folds[@]}"
do
  python F:/code/myProjects/dlkt/example/prepare_dataset/dataset_debias_method2.py \
    --max_seq_len2delete 200 --acc_th 0.28 \
    --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/ednet-kt1_train_fold_${fold}.txt"
done