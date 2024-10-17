#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/lpkt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 15 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 32 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --ablation_set 2 --dim_e 64 --dim_k 64 --dim_correct 50 --dropout 0.2 \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done
} >> F:/code/myProjects/dlkt/example/result_local/lpkt_our_setting_new_assist2009_save.txt