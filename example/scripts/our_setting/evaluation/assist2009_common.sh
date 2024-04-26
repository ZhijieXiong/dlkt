#!/usr/bin/env bash


# 只测3种情况下偏差
previous_seq_lens=(20 30 40)
seqs_most_acc=("0.4" "0.3" "0.2")


for i in `seq 0 2`;
do
  previous_seq_len4bias=${previous_seq_lens[$i]}
  seq_most_accuracy4bias=${seqs_most_acc[$i]}

  echo -e "(${previous_seq_len4bias}, ${seq_most_accuracy4bias})"
  python F:/code/myProjects/dlkt/example/evaluate.py \
    --debug_mode False --use_cpu False \
    --save_model_dir "" \
    --save_model_name "saved.ckt" --model_name_in_ckt "best_valid" \
    --setting_name "our_setting" --dataset_name "assist2009" --test_file_name "assist2009_test_fold_0.txt" \
    --data_type "only_question" --base_type "concept" --evaluate_batch_size 256 \
    --train_statics_common_path "F:\code\myProjects\dlkt\lab\settings\our_setting\assist2009_train_fold_0_statics_common.json" \
    --train_statics_special_path "F:\code\myProjects\dlkt\lab\settings\our_setting\assist2009_train_fold_0_statics_special.json" \
    --max_seq_len 200 --seq_len_absolute "[0, 10, 100, 200]" \
    --previous_seq_len4bias "${previous_seq_len4bias}" --seq_most_accuracy4bias "${seq_most_accuracy4bias}" \
    --transfer_head2zero False --is_dimkt False  --is_dct False
done
