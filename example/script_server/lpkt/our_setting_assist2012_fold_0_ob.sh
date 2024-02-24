#!/usr/bin/env bash

{
  dataset_name="assist2012"
  data_type="single_concept"
  fold=0


  weight_decays='0.00001 0.000001'
  dropouts='0.2 0.3 0.4'
  for weight_decay in ${weight_decays}
  do
    for dropout in ${dropouts}
    do
      echo -e "weight_decay: ${weight_decay}, dropout: ${dropout}"
      CUDA_VISIBLE_DEVICES=1 python /home/xiongzj/myProjects/KT/dlkt/example/train/lpkt.py \
        --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
        --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
        --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
        --train_strategy "valid_test" --num_epoch 200 \
        --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
        --main_metric "AUC" --use_multi_metrics False \
        --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 15 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
        --train_batch_size 64 --evaluate_batch_size 256 \
        --enable_clip_grad False --grad_clipped 10.0 \
        --num_concept 265 --num_question 53091 \
        --dim_e 128 --dim_k 128 --dim_correct 50 --dropout "${dropout}" \
        --save_model False --debug_mode False --use_cpu False --seed 0
    done
  done
} >> /home/xiongzj/myProjects/KT/dlkt/example/results/lpkt_our_setting_assist2012_fold_0_ob.txt