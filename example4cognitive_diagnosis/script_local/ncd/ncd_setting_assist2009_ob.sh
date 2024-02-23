#!/usr/bin/env bash

{
  setting_name="ncd_setting"
  dataset_name="assist2009"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example4cognitive_diagnosis/train/ncd.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type adam --weight_decay 0 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_user 2500 --num_concept 123 --num_question 17751 --dim_predict1 512 --dim_predict2 256 --dropout 0.5 \
      --save_model False --use_cpu False --debug_mode False --seed 0
  done
} >> F:/code/myProjects/dlkt/example4cognitive_diagnosis/result_local/ncd_ncd_setting_assist2009.txt