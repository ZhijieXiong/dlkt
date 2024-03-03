#!/usr/bin/env bash

{
  dataset_name="assist2017"
  data_type="single_concept"
  fold=0

  dropouts='0.05 0.1 0.15 0.2 0.25'
  dims_model=(256 512)
  grads_clipped='1 10'
  learning_rates='0.0001'
  for learning_rate in ${learning_rates}
  do
    for grad_clipped in ${grads_clipped}
    do
      for num_block in 1 2
      do
        for dim_model in "${dims_model[@]}"
        do
          for dropout in ${dropouts}
          do
            echo -e "learning_rate: ${learning_rate}, grad_clipped: ${grad_clipped}, num_block: ${num_block}, dim_model: ${dim_model}, dropout: ${dropout}"
            python /ghome/xiongzj/code/dlkt/example/train/akt.py \
              --setting_name "akt_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
              --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
              --train_strategy "valid_test" --num_epoch 300 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric "AUC" --use_multi_metrics False \
              --learning_rate "${learning_rate}" --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
              --train_batch_size 24 --evaluate_batch_size 128 \
              --enable_clip_grad True --grad_clipped "${grad_clipped}" \
              --num_concept 101 --num_question 2803 \
              --dim_model "${dim_model}" --key_query_same True --num_head 8 --num_block "${num_block}" --dim_ff "${dim_model}" --dim_final_fc 512 --dropout "${dropout}" \
              --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
              --save_model False --debug_mode False --use_cpu False --seed 0
          done
        done
      done
    done
  done


#  dropouts='0.05 0.1 0.15 0.2 0.25'
#  dims_model=(256 512)
#  learning_rates='0.0001'
#  for learning_rate in ${learning_rates}
#  do
#    for num_block in 1 2
#    do
#      for dim_model in "${dims_model[@]}"
#      do
#        for dropout in ${dropouts}
#        do
#          echo -e "learning_rate: ${learning_rate}, grad_clipped: False, num_block: ${num_block}, dim_model: ${dim_model}, dropout: ${dropout}"
#          python /ghome/xiongzj/code/dlkt/example/train/akt.py \
#            --setting_name "akt_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
#            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
#            --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#            --train_strategy "valid_test" --num_epoch 300 \
#            --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#            --main_metric "AUC" --use_multi_metrics False \
#            --learning_rate "${learning_rate}" --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#            --train_batch_size 24 --evaluate_batch_size 128 \
#            --enable_clip_grad False --grad_clipped 0 \
#            --num_concept 101 --num_question 2803 \
#            --dim_model "${dim_model}" --key_query_same True --num_head 8 --num_block "${num_block}" --dim_ff "${dim_model}" --dim_final_fc 512 --dropout "${dropout}" \
#            --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
#            --save_model False --debug_mode False --use_cpu False --seed 0
#        done
#      done
#    done
#  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/akt_akt_setting_assist2017_fold_0_ob.txt