#!/usr/bin/env bash

{
  dataset_name="bridge2algebra2006"
  data_type="only_question"
  fold=0

  dropouts='0.1 0.2 0.3'
  for dim_model in 256 512
  do
    for num_block in 1 2
    do
      for num_head in 4 8
      do
        for dropout in ${dropouts}
        do
          echo -e "dim_model: ${dim_model}, num_block: ${num_block}, num_head: ${num_head}, dropout: ${dropout}"
          CUDA_VISIBLE_DEVICES=0 python /home/xiongzj/myProjects/KT/dlkt/example/train/akt.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type adam --weight_decay 0 --momentum 0.9 \
            --train_strategy valid_test --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
            --main_metric AUC --use_multi_metrics False \
            --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 24 --evaluate_batch_size 128 \
            --enable_clip_grad True --grad_clipped 10.0 \
            --num_concept 493 --num_question 129263 \
            --dim_model "${dim_model}" --key_query_same True --num_head "${num_head}" --num_block "${num_block}" --dim_ff "${dim_model}" --dim_final_fc 512 --dropout "${dropout}" \
            --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
            --save_model True --seed 0
        done
      done
    done
  done
} >> /home/xiongzj/myProjects/KT/dlkt/example/results/akt_our_setting_bridge2algebra2006_fold_0_ob.txt