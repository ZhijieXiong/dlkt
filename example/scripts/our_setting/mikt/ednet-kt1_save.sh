#!/usr/bin/env bash

{
  dataset_name="ednet-kt1"
  data_type="only_question"
  num_concept=188
  num_question=11858

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python /ghome/xiongzj/code/dlkt/example/train/mikt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 80 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 15.0 \
      --num_concept "${num_concept}" --num_question "${num_question}" --dim_emb 64 --dim_state 64 --dropout 0.2 --seq_len 200 \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/mikt_our_setting_new_ednet-kt1_save.txt