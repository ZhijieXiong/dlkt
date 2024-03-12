#!/usr/bin/env bash

{
  dataset_name="edi2020-task34"
  data_type="single_concept"
  fold=0

  dropouts='0.05 0.1 0.15'
  weight_decays='0.001 0.0001 0.00001 0.000001'
  nums_rnn_layer=(1 2)
  dims_question=(32 64)
  dims_latent=(32 64)
  for weight_decay in ${weight_decays}
  do
    for num_rnn_layer in "${nums_rnn_layer[@]}"
      do
      for dim_question in "${dims_question[@]}"
        do
          for dim_latent in "${dims_latent[@]}"
          do
            for dropout in ${dropouts}
            do
              echo -e "weight_decay: ${weight_decay}, num_rnn_layer: ${num_rnn_layer}, dim_question: ${dim_question}, dim_latent: ${dim_latent}, dropout: ${dropout}"
              python F:/code/myProjects/dlkt/example/train/dct.py \
                --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                --train_strategy "valid_test" --num_epoch 200 \
                --use_early_stop True --epoch_early_stop 15 --use_last_average False --epoch_last_average 5 \
                --main_metric "AUC" --use_multi_metrics False \
                --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                --train_batch_size 64 --evaluate_batch_size 256 \
                --enable_clip_grad False --grad_clipped 10.0 \
                --num_concept 53 --num_question 948 \
                --multi_stage True --test_theory "irt" \
                --dim_question "${dim_question}" --dim_correct 64 --dim_latent "${dim_latent}" --rnn_type "gru" \
                --num_rnn_layer "${num_rnn_layer}" --que_user_share_proj False --dropout "${dropout}" \
                --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 --w_q_table 0 \
                --save_model False --debug_mode False --use_cpu False --seed 0
          done
        done
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/dct_irt_not_share_baseline_our_setting_edi2020-task34_fold_0_ob.txt