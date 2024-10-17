#!/usr/bin/env bash

{
  dataset_name="xes3g5m"
  data_type="only_question"
  num_concept=865
  num_question=7652

  folds=(0 1 2 3 4)

  for fold in "${folds[@]}"
  do
    python /ghome/xiongzj/code/dlkt/example/train/sparse_kt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept "${num_concept}" --num_question "${num_question}" \
      --dim_model 256 --num_block 2 --num_head 8 --dim_ff 256 \
      --dim_final_fc 256 --dim_final_fc2 256 --dropout 0.2 --seq_len 200 --kq_same True --separate_qa False \
      --difficulty_scalar False --k_index 5 --use_akt_rasch True \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done

} >> /ghome/xiongzj/code/dlkt/example/result_cluster/sparse_kt_our_setting_new_xes3g5m_save.txt