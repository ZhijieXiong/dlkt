#!/usr/bin/env bash

{
  dataset_name="assist2017"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/qikt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type adam --weight_decay 0 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 32 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 101 --num_question 2803 \
      --dim_emb 256 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 --num_mlp_layer 2 \
      --lambda_q_all 1 --lambda_c_next 1 --lambda_c_all 1 --use_irt True \
      --weight_predict_q_all_loss 2 --weight_predict_q_next_loss 1 \
      --weight_predict_c_all_loss 1 --weight_predict_c_next_loss 1 \
      --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done
} >> F:/code/myProjects/dlkt/example/result_local/qikt_our_setting_new_assist2017_save.txt