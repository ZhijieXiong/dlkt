#!/usr/bin/env bash

previous_seq_lens=(20 30 40)
seqs_most_acc='0.4 0.3 0.2'


for previous_seq_len4bias in "${previous_seq_lens[@]}"
do
  for seq_most_accuracy4bias in ${seqs_most_acc}
  do
    echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
    python F:/code/myProjects/dlkt/example/evaluate.py \
      --debug_mode False --use_cpu False \
      --save_model_dir "F:\code\myProjects\dlkt\lab\saved_models\2024-03-27@14-10-34@@DCT@@seed_0@@our_setting_new@@assist2017_train_fold_0" \
      --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
      --setting_name "our_setting_new" --dataset_name "assist2017" --test_file_name "assist2017_test_fold_0.txt" \
      --data_type "single_concept" --base_type "concept" --evaluate_batch_size 128 \
      --statics_file_path "F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2017_train_fold_0_statics.json" \
      --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
      --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
      --transfer_head2zero False --is_dimkt False
  done
done