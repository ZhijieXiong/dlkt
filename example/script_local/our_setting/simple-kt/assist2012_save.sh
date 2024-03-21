#!/usr/bin/env bash

{
  dataset_name="assist2012"
  data_type="single_concept"
  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/simple_kt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0004 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 265 --num_question 53091 \
      --dim_model 256 --num_block 2 --num_head 4 --dim_ff 256 \
      --dim_final_fc 512 --dim_final_fc2 256 --dropout 0.1 \
      --seq_len 200 --key_query_same True --separate_qa False --difficulty_scalar False \
      --use_sample_weight False --sample_weight_method "highlight_tail" --tail_weight 1.1 \
      --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/simple_kt_our_setting_assist2012_save.txt