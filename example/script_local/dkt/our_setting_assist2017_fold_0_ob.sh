#!/usr/bin/env bash

{
  setting_name="our_setting"
  dataset_name="assist2017"
  data_type="single_concept"
  fold=0

  dropouts='0.1 0.2 0.3 0.5'
  weight_decays='0 0.001 0.0001'
  for weight_decay in ${weight_decays}
  do
    for dim_emb in 64 256
    do
      for latent in 64 256
      do
        for num_predict_layer in 1 2
        do
          for dim_predict_mid in 256 512
          do
            for dropout in ${dropouts}
            do
              echo -e "wight_decay: ${weight_decay}, dim_emb: ${dim_emb}, latent: ${latent}, num_predict_layer: ${num_predict_layer}, dim_predict_mid: ${dim_predict_mid}, dropout: ${dropout}"
              python F:/code/myProjects/dlkt/example/train/dkt.py \
                --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                --optimizer_type adam --weight_decay "${weight_decay}" --momentum 0.9 \
                --train_strategy valid_test --num_epoch 200 \
                --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                --main_metric AUC --use_multi_metrics False \
                --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                --train_batch_size 64 --evaluate_batch_size 128 \
                --enable_clip_grad False --grad_clipped 10.0 \
                --use_concept True --num_concept 101 --num_question 2803 \
                --dim_emb "${dim_emb}" --dim_latent "${latent}" --rnn_type gru --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer "${num_predict_layer}" --dim_predict_mid "${dim_predict_mid}" --activate_type sigmoid \
                --save_model False --seed 0
            done
          done
        done
      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/dkt_our_setting_assist2017_fold_0_ob.txt