#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"
  folds=(0 1 2 3 4)
  # weight_decay: 0.000001, weight_aux_emb: 1, dim_question: 64, dim_latent: 128, num_rnn_layer: 1, num_predict_layer: 2, dropout: 0.3
  for fold in "${folds[@]}"
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/aux_info_qdkt.py \
      --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type adam --weight_decay 0.000001 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --weight_aux_emb 1 --dim_question 64 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 \
      --num_predict_layer 2 --dim_predict_mid 128 --activate_type "relu" \
      --pretrain_aux_emb_path "F:\code\myProjects\dlkt\lab\saved_models\save\our_setting_new\AuxInfoQDKT\no_pretrain_aux_emb\2024-03-26@14-55-23@@AuxInfoQDKT@@seed_0@@our_setting_new@@assist2017_train_fold_0\saved.ckt" \
      --save_model False --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/aux-info-qdkt-w-pretrain-aux-emb_our_setting_new_assist2009_save.txt