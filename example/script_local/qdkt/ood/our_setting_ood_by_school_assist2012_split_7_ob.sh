#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="assist2012"
  data_type="single_concept"
  split=7


  dropouts='0.1 0.2 0.3'
  weight_decays='0.001 0.0001 0'
  for weight_decay in ${weight_decays}
  do
    for dim_correct in 64 128
      do
        for dim_latent in 64 128
        do
          for dropout in ${dropouts}
          do
            echo -e "weight_decay: ${weight_decay}, dim_correct: ${dim_correct}, dim_latent: ${dim_latent}, dropout: ${dropout}"
            python F:/code/myProjects/dlkt/example/train/qdkt.py \
              --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
              --optimizer_type adam --weight_decay "${weight_decay}" --momentum 0.9 \
              --train_strategy valid_test --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric AUC --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
              --train_batch_size 64 --evaluate_batch_size 256 \
              --enable_clip_grad False --grad_clipped 10.0 \
              --num_concept 265 --num_question 53091 \
              --dim_concept 64 --dim_question 64 --dim_correct "${dim_correct}" --dim_latent "${dim_latent}" --rnn_type gru --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer 3 --dim_predict_mid 128 --activate_type relu \
              --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
              --transfer_head2zero False --head2tail_transfer_method "mean_pool" \
              --save_model False --seed 0
        done
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/qdkt_our_setting_ood_by_school_assist2012_split_7_ob.txt