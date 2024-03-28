#!/usr/bin/env bash


{
  dataset_name="assist2009"
  data_type="only_question"
  fold=0

  for lr_schedule_step in 1 2 3 4 5
  do
    echo -e "lr_schedule_step: ${lr_schedule_step}"
    python /ghome/xiongzj/code/dlkt/example/train/lbkt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --num_concept 123 --num_question 17751 \
      --learning_rate 0.002 --enable_lr_schedule True \
      --lr_schedule_type "StepLR" --lr_schedule_step "${lr_schedule_step}" --lr_schedule_gamma 0.5 \
      --save_model False --debug_mode False --use_cpu False --seed 0
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/lbkt_our_setting_new_assist2009_fold_0_ob.txt
