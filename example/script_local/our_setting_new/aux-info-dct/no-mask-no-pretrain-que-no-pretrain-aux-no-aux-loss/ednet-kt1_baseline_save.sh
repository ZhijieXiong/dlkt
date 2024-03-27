#!/usr/bin/env bash

{
  dataset_name="ednet-kt1"
  data_type="only_question"
  folds=(0 1 2 3 4)

  # weight_decay: 0.00001, weight_aux_emb: 0.5, dim_question: 64, dim_latent: 128, num_rnn_layer: 3, num_mlp_layer: 1, dropout: 0.1
  for fold in "${folds[@]}"
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/aux_info_dct.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 188 --num_question 11858 \
      --test_theory "irt" --que_user_share_proj False \
      --weight_aux_emb 0.5 --dim_question 64 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 3 --num_mlp_layer 1 --dropout 0.1 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/aux-info-dct_not-share_baseline_our_setting_new_ednet-kt1_save.txt