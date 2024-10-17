#!/usr/bin/env bash


{
  dataset_name="junyi2015"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/lbkt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 3 --lr_schedule_gamma 0.5 \
      --train_batch_size 32 --evaluate_batch_size 128 \
      --num_concept 40 --num_question 817 \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done
} >> F:/code/myProjects/dlkt/example/result_local/lbkt_our_setting_new_junyi2015_save.txt
