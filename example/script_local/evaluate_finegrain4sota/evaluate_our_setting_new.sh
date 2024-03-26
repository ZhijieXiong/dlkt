#!/usr/bin/env bash


previous_seq_lens=(20 30 40)
seqs_most_acc='0.4 0.3 0.2'

# assist2009: AKT
for previous_seq_len4bias in "${previous_seq_lens[@]}"
do
  for seq_most_accuracy4bias in ${seqs_most_acc}
  do
    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
    python F:/code/myProjects/dlkt/example/evaluate.py \
      --debug_mode False --use_cpu False \
      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\AKT\2024-03-22@12-34-53@@AKT@@seed_0@@our_setting_new@@assist2009_train_fold_0" \
      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
      --setting_name "our_setting_new" --dataset_name "assist2009" --test_file_name "assist2009_test_fold_0.txt" \
      --data_type "only_question" --base_type "concept" --evaluate_batch_size 128 \
      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0_statics.json" \
      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
      --transfer_head2zero False --is_dimkt False
  done
done

# assist2012: AKT
for previous_seq_len4bias in "${previous_seq_lens[@]}"
do
  for seq_most_accuracy4bias in ${seqs_most_acc}
  do
    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
    python F:/code/myProjects/dlkt/example/evaluate.py \
      --debug_mode False --use_cpu False \
      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\AKT\2024-03-22@13-06-17@@AKT@@seed_0@@our_setting_new@@assist2012_train_fold_0" \
      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
      --setting_name "our_setting_new" --dataset_name "assist2012" --test_file_name "assist2012_test_fold_0.txt" \
      --data_type "single_concept" --base_type "concept" --evaluate_batch_size 128 \
      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2012_train_fold_0_statics.json" \
      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
      --transfer_head2zero False --is_dimkt False
  done
done

# assist2017: LBKT


# junyi2015: LBKT


## ednet-kt1: LPKT
#for previous_seq_len4bias in "${previous_seq_lens[@]}"
#do
#  for seq_most_accuracy4bias in ${seqs_most_acc}
#  do
#    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
#    python F:/code/myProjects/dlkt/example/evaluate.py \
#      --debug_mode False --use_cpu False \
#      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\LPKT\2024-03-23@10-54-32@@LPKT@@seed_0@@our_setting_new@@ednet-kt1_train_fold_0" \
#      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
#      --setting_name "our_setting_new" --dataset_name "ednet-kt1" --test_file_name "ednet-kt1_test_fold_0.txt" \
#      --data_type "only_question" --base_type "concept" --evaluate_batch_size 128 \
#      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\ednet-kt1_train_fold_0_statics.json" \
#      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
#      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
#      --transfer_head2zero False --is_dimkt False
#  done
#done
#
## slepemapy: LPKT
#for previous_seq_len4bias in "${previous_seq_lens[@]}"
#do
#  for seq_most_accuracy4bias in ${seqs_most_acc}
#  do
#    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
#    python F:/code/myProjects/dlkt/example/evaluate.py \
#      --debug_mode False --use_cpu False \
#      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\LPKT\2024-03-23@23-23-55@@LPKT@@seed_0@@our_setting_new@@slepemapy_train_fold_0" \
#      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
#      --setting_name "our_setting_new" --dataset_name "slepemapy" --test_file_name "slepemapy_test_fold_0.txt" \
#      --data_type "single_concept" --base_type "concept" --evaluate_batch_size 128 \
#      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\slepemapy_train_fold_0_statics.json" \
#      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
#      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
#      --transfer_head2zero False --is_dimkt False
#  done
#done

## statics2011: QIKT
#for previous_seq_len4bias in "${previous_seq_lens[@]}"
#do
#  for seq_most_accuracy4bias in ${seqs_most_acc}
#  do
#    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
#    python F:/code/myProjects/dlkt/example/evaluate.py \
#      --debug_mode False --use_cpu False \
#      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\QIKT\2024-03-22@10-15-18@@QIKT@@seed_0@@our_setting_new@@statics2011_train_fold_0" \
#      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
#      --setting_name "our_setting_new" --dataset_name "statics2011" --test_file_name "statics2011_test_fold_0.txt" \
#      --data_type "single_concept" --base_type "concept" --evaluate_batch_size 128 \
#      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\statics2011_train_fold_0_statics.json" \
#      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
#      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
#      --transfer_head2zero False --is_dimkt False
#  done
#done
#
## edi2020-task34: QIKT
#for previous_seq_len4bias in "${previous_seq_lens[@]}"
#do
#  for seq_most_accuracy4bias in ${seqs_most_acc}
#  do
#    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
#    python F:/code/myProjects/dlkt/example/evaluate.py \
#      --debug_mode False --use_cpu False \
#      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\QIKT\2024-03-22@09-24-31@@QIKT@@seed_0@@our_setting_new@@edi2020-task34_train_fold_0" \
#      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
#      --setting_name "our_setting_new" --dataset_name "edi2020-task34" --test_file_name "edi2020-task34_test_fold_0.txt" \
#      --data_type "single_concept" --base_type "concept" --evaluate_batch_size 128 \
#      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\edi2020-task34_train_fold_0_statics.json" \
#      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
#      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
#      --transfer_head2zero False --is_dimkt False
#  done
#done

