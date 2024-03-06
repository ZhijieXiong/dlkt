#!/usr/bin/env bash

dataset_name="ednet-kt1"
data_type="only_question"
folds=(0 1 2 3 4)

{
  # baseline
  # weight_decay: 0.0001, num_rnn_layer: 2, dim_question: 64, dim_latent: 64, dropout: 0.05
  for fold in "${folds[@]}"
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dct.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 188 --num_question 11858 \
      --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 2 --dropout 0.05 \
      --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dct_baseline_our_setting_ednet-kt1_save.txt