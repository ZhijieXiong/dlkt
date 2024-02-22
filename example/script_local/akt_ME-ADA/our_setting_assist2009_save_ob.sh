#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"

#  for fold in 0 1 2 3 4
#  do
#    echo -e "fold: ${fold}"
#    python F:/code/myProjects/dlkt/example/train/akt_max_entropy_aug.py \
#      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
#      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 \
#      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False \
#      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --train_batch_size 24 --evaluate_batch_size 128 \
#      --enable_clip_grad True --grad_clipped 10.0 \
#      --num_concept 123 --num_question 17751 \
#      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
#      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
#      --use_warm_up True --epoch_warm_up 1 \
#      --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss 0.5 --loop_adv 3 --adv_learning_rate 1 --eta 20 --gamma 20 \
#      --save_model False --debug_mode False --seed 0
#  done

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt_max_entropy_aug.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --use_warm_up True --epoch_warm_up 2 \
      --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss 0.5 --loop_adv 3 --adv_learning_rate 1 --eta 20 --gamma 20 \
      --save_model False --debug_mode False --seed 0
  done

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt_max_entropy_aug.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --use_warm_up True --epoch_warm_up 2 \
      --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss 0.5 --loop_adv 3 --adv_learning_rate 1 --eta 20 --gamma 20 \
      --save_model False --debug_mode False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_ME-ADA_our_setting_assist2009_save_ob.txt