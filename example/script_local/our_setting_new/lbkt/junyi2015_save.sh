#!/usr/bin/env bash


{
  dataset_name="junyi2015"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/lbkt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --num_concept 40 --num_question 817 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/lbkt_our_setting_new_junyi2015_save.txt
