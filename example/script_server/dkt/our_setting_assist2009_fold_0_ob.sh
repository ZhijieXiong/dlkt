#!/usr/bin/env bash

{
  dataset_name="assist2009"
  fold=0

  dropouts='0.05 0.1 0.15 0.2 0.25'
  lrs='0.001 0.0001'
  for lr in ${lrs}
  do
    for dim_emb in 256 512
    do
      for latent in 256 512
      do
        for num_predict_layer in 1 2 3
        do
          for dim_predict_mid in 256 512
          do
            for dropout in ${dropouts}
            do
              echo -e "lr: ${lr}, dim_emb: ${dim_emb}, latent: ${latent}, num_predict_layer: ${num_predict_layer}, dim_predict_mid: ${dim_predict_mid}, dropout: ${dropout}"
              python F:/code/myProjects/dlkt/example/train/dkt.py \
                --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "only_question" \
                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                --optimizer_type adam --weight_decay 0 --momentum 0.9 \
                --train_strategy valid_test --num_epoch 200 \
                --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                --main_metric AUC --use_multi_metrics False \
                --learning_rate "${lr}" --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                --train_batch_size 64 --evaluate_batch_size 128 \
                --enable_clip_grad False --grad_clipped 10.0 \
                --use_concept True --num_concept 123 --num_question 17751 \
                --dim_emb "${dim_emb}" --dim_latent "${latent}" --rnn_type gru --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer "${num_predict_layer}" --dim_predict_mid "${dim_predict_mid}" --activate_type sigmoid \
                --save_model False --seed 0
            done
          done
        done
      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/dkt_our_setting_assist2009_fold_0_ob.txt