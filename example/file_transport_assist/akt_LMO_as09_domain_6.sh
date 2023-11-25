#!/bin/bash

{
  seeds=(0 1 2 3 4)
  dataset_name="assist2009"

  for seed in "${seeds[@]}"
  do
    echo -e "seed: ${seed}"
    python /ghome/xiongzj/code/dlkt/example/train/akt.py \
      --setting_name "random_split_leave_multi_out_setting" --data_type "multi_concept" \
      --train_file_name "${dataset_name}_train_split_6.txt" --valid_file_name "${dataset_name}_valid_split_6.txt" --test_file_name "${dataset_name}_test_split_6.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 128 \
      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --enable_clip_grad False --grad_clipped 10 \
      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
      --save_model False --seed "${seed}"
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/baseline/save/akt_LMO_as09_domain_6.txt
