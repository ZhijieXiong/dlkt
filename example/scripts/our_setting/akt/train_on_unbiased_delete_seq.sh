#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
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
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_unbiased_by_delete_seq_our_setting_new_assist2009_save.txt


{
  dataset_name="assist2012"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 265 --num_question 53091 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 3 --dim_ff 256 --dim_final_fc 512 --dropout 0.2 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_unbiased_by_delete_seq_our_setting_new_assist2012_save.txt


{
  dataset_name="assist2017"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 101 --num_question 2803 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_unbiased_by_delete_seq_our_setting_new_assist2017_save.txt


{
  dataset_name="ednet-kt1"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 188 --num_question 11858 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 1 --dim_ff 256 --dim_final_fc 512 --dropout 0.25 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_unbiased_by_delete_seq_our_setting_new_ednet-kt1_save.txt


{
  dataset_name="statics2011"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}_unbiased_by_delete_seq.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 27 --num_question 1223 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_unbiased_by_delete_seq_our_setting_new_statics2011_save.txt