#!/usr/bin/env bash

setting_name="pykt_question_setting"


{
  dataset_name="algebra2005"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 112 --num_question 173113 \
      --dim_model 512 --key_query_same True --num_head 4 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.3 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_algebra2005_save.txt


{
  dataset_name="assist2009"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
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

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_assist2009_save.txt


{
  dataset_name="bridge2algebra2006"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 493 --num_question 129263 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_bridge2algebra2006_save.txt


{
  dataset_name="edi2020-task34"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 53 --num_question 948 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.1 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_edi2020-task34_save.txt


{
  dataset_name="statics2011"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
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

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_statics2011_save.txt


{
  dataset_name="xes3g5m"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_concept 865 --num_question 7652 \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 4 --dim_ff 256 --dim_final_fc 512 --dropout 0.2 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt_pykt_question_setting_xes3g5m_save.txt


{
  setting_name="our_setting_ood_by_country"
  dataset_name="slepemapy"
  data_type="single_concept"
  split=0

  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 246 --num_question 5730 --model_version 2 --ablation_set 0 \
      --dim_e 48 --dim_k 48 --dim_correct 50 --dropout 0.2 \
      --save_model True --debug_mode False --use_cpu False --seed "${seed}"
  done

} >> F:/code/myProjects/dlkt/example/result_local/lpkt_plus_v2_our_setting_ood_slepemapy_split_0_save_ob.txt


{
  setting_name="our_setting_ood_by_school"
  dataset_name="assist2012"
  data_type="single_concept"
  split=7

  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
      --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 265 --num_question 53091 --model_version 2 --ablation_set 0 \
      --dim_e 64 --dim_k 64 --dim_correct 50 --dropout 0.1 \
      --save_model False --debug_mode False --use_cpu False --seed "${seed}"
  done
} >> F:/code/myProjects/dlkt/example/result_local/lpkt_plus_v2_our_setting_ood_assist2012_split_7_save_ob.txt