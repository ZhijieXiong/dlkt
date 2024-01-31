#!/usr/bin/env bash

{
  dataset_name="bridge2algebra2006"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/qdkt.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type adam --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 493 --num_question 129263 \
      --dim_concept 64 --dim_question 64 --dim_correct 128 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.4 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
      --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
      --save_model True --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/qdkt_same_as_ME-ADA_our_setting_bridge2algebra2006_save.txt