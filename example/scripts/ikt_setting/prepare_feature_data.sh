#!/usr/bin/env bash


dataset_name="assist2009"
for fold in 0 1 2 3 4
do
  python F:/code/myProjects/dlkt/example/ikt_feature_engineer.py \
    --setting_name "ikt_setting" --dataset_name "${dataset_name}" --data_type "only_question" \
    --train_file_name "${dataset_name}_all.txt" \
    --valid_file_name "${dataset_name}_train_fold_${fold}.txt" \
    --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
    --num_concept 123 --num_question 17751 \
    --num_cluster 7 --ability_evaluate_interval 20 --num_min_question 4 --num_question_diff 11 \
    --min_seq_len 20 --seed 0
done


dataset_name="assist2012"
for fold in 0 1 2 3 4
do
  python F:/code/myProjects/dlkt/example/ikt_feature_engineer.py \
    --setting_name "ikt_setting" --dataset_name "${dataset_name}" --data_type "single_concept" \
    --train_file_name "${dataset_name}_all.txt" \
    --valid_file_name "${dataset_name}_train_fold_${fold}.txt" \
    --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
    --num_concept 265 --num_question 53091 \
    --num_cluster 7 --ability_evaluate_interval 20 --num_min_question 4 --num_question_diff 11 \
    --min_seq_len 20 --seed 0
done


dataset_name="algebra2005"
for fold in 0 1 2 3 4
do
  python F:/code/myProjects/dlkt/example/ikt_feature_engineer.py \
    --setting_name "ikt_setting" --dataset_name "${dataset_name}" --data_type "only_question" \
    --train_file_name "${dataset_name}_all.txt" \
    --valid_file_name "${dataset_name}_train_fold_${fold}.txt" \
    --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
    --num_concept 112 --num_question 173113 \
    --num_cluster 7 --ability_evaluate_interval 20 --num_min_question 4 --num_question_diff 11 \
    --min_seq_len 20 --seed 0
done