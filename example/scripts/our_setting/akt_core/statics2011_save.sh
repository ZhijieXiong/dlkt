#!/usr/bin/env bash

{
  dataset_name="statics2011"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/akt_core.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 128 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 27 --num_question 1223 \
      --dim_model 64 --dim_ff 64 --dim_final_fc 64 --key_query_same True --separate_qa False \
      --num_block 2 --num_head 4 --dropout 0.2 --fusion_mode "rubin" --weight_rasch_loss 0.00001 \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done

} >> F:/code/myProjects/dlkt/example/result_local/akt-core_our_setting_new_statics2011_save.txt