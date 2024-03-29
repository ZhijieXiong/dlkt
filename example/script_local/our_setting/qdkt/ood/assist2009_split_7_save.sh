#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="assist2009"
  data_type="only_question"
  split=7


  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/qdkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
      --optimizer_type adam --weight_decay 0.001 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type gru --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 3 --dim_predict_mid 128 --activate_type relu \
      --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
      --save_model True --debug_mode False --use_cpu False --seed "${seed}"
  done

} >> F:/code/myProjects/dlkt/example/result_local/qdkt_our_setting_ood_by_school_assist2009_split_7_save.txt