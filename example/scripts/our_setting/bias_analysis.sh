#!/usr/bin/env bash


setting_name="our_setting_new"
dataset_names=("assist2009" "assist2012" "assist2017" "ednet-kt1" "statics2011")
# "slepemapy" "junyi2015" "edi2020-task34" "edi2020-task1"
folds=(0 1 2 3 4)

#{
#  for dataset_name in "${dataset_names[@]}"
#  do
#    for fold in "${folds[@]}"
#    do
#      echo -e "------------------------------------fold_${fold}-----------------------------------------"
#      python F:/code/myProjects/dlkt/example/data_analysis/biased_analysis.py \
#        --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/${dataset_name}_train_fold_${fold}.txt"
#      echo -e "------------------------------------------------------------------------------------------\n"
#    done
#  done
#} >> F:/code/myProjects/dlkt/example/result_local/train_dataset_bias_analysis_result.txt
#
#
#
#{
#  for dataset_name in "${dataset_names[@]}"
#  do
#    for fold in "${folds[@]}"
#    do
#      echo -e "------------------------------------fold_${fold}-----------------------------------------"
#      python F:/code/myProjects/dlkt/example/data_analysis/biased_analysis.py \
#        --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/${dataset_name}_test_fold_${fold}.txt"
#      echo -e "------------------------------------------------------------------------------------------\n"
#    done
#  done
#} >> F:/code/myProjects/dlkt/example/result_local/test_dataset_bias_analysis_result.txt




{
  for dataset_name in "${dataset_names[@]}"
  do
    for fold in "${folds[@]}"
    do
      echo -e "------------------------------------fold_${fold}-----------------------------------------"
      python F:/code/myProjects/dlkt/example/data_analysis/biased_analysis.py \
        --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/${dataset_name}_train_fold_${fold}_unbiased_by_delete_sample.txt"
      echo -e "------------------------------------------------------------------------------------------\n"
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/unbiased_train_dataset_by_delete_sample_bias_analysis_result.txt


#{
#  for dataset_name in "${dataset_names[@]}"
#  do
#    for fold in "${folds[@]}"
#    do
#      echo -e "------------------------------------fold_${fold}-----------------------------------------"
#      python F:/code/myProjects/dlkt/example/data_analysis/biased_analysis.py \
#        --data_path "F:/code/myProjects/dlkt/lab/settings/${setting_name}/${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt"
#      echo -e "------------------------------------------------------------------------------------------\n"
#    done
#  done
#} >> F:/code/myProjects/dlkt/example/result_local/unbiased_train_dataset_by_delete_seq_bias_analysis_result.txt