#!/usr/bin/env bash

python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "assist2009" --data_type "only_question" \
  --num_concept 123 --num_question 17751 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 35 --question_acc_middle_high 80 \
  --concept_acc_low_middle 35 --concept_acc_middle_high 80


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\ednet-kt1_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "ednet-kt1" --data_type "only_question" \
  --num_concept 188 --num_question 11858 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 35 --question_acc_middle_high 80 \
  --concept_acc_low_middle 35 --concept_acc_middle_high 80


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\xes3g5m_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "xes3g5m" --data_type "only_question" \
  --num_concept 865 --num_question 7652 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 40 --question_acc_middle_high 90 \
  --concept_acc_low_middle 40 --concept_acc_middle_high 90


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2012_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "assist2012" --data_type "single_concept" \
  --num_concept 265 --num_question 53091 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 35 --question_acc_middle_high 85 \
  --concept_acc_low_middle 35 --concept_acc_middle_high 85


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2017_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "assist2017" --data_type "single_concept" \
  --num_concept 101 --num_question 2803 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 20 --question_acc_middle_high 65 \
  --concept_acc_low_middle 20 --concept_acc_middle_high 65


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\edi2020-task34_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "edi2020-task34" --data_type "single_concept" \
  --num_concept 53 --num_question 948 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 30 --question_acc_middle_high 70 \
  --concept_acc_low_middle 30 --concept_acc_middle_high 70


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\edi2020-task1_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "edi2020-task1" --data_type "single_concept" \
  --num_concept 282 --num_question 27613 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 35 --question_acc_middle_high 80 \
  --concept_acc_low_middle 35 --concept_acc_middle_high 80


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\statics2011_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "statics2011" --data_type "single_concept" \
  --num_concept 27 --num_question 1223 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 40 --question_acc_middle_high 85 \
  --concept_acc_low_middle 40 --concept_acc_middle_high 85


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\junyi2015_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "junyi2015" --data_type "single_concept" \
  --num_concept 40 --num_question 817 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 40 --question_acc_middle_high 90 \
  --concept_acc_low_middle 40 --concept_acc_middle_high 90


python F:/code/myProjects/dlkt/example/prepare4fine_trained_evaluate.py \
  --target_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\slepemapy_train_fold_0.txt" \
  --preprocessed_dir "F:\code\myProjects\dlkt\lab\dataset_preprocessed" \
  --dataset_name "slepemapy" --data_type "single_concept" \
  --num_concept 246 --num_question 5730 \
  --use_absolute4fre False \
  --question_fre_percent_lowest 0.2 --question_fre_percent_highest 0.8 \
  --concept_fre_percent_lowest 0.2 --concept_fre_percent_highest 0.8 \
  --use_absolute4acc True \
  --question_acc_low_middle 40 --question_acc_middle_high 85 \
  --concept_acc_low_middle 40 --concept_acc_middle_high 85