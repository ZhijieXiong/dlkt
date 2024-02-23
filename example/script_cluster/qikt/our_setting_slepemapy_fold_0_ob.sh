#!/usr/bin/env bash

{
  dataset_name="slepemapy"
  data_type="single_concept"
  fold=0

  dropouts='0.3 0.5'
  weight_decays='0'
  lambdas_q_all='1'
  lambdas_c_next='1'
  lambdas_c_all='1'
  qs_all_loss='1 2'
  cs_all_loss='1 2'
  cs_next_loss='1 2'
  nums_mlp_layer=(1 2)
  for weight_decay in ${weight_decays}
  do
    for lambda_q_all in ${lambdas_q_all}
    do
      for lambda_c_next in ${lambdas_c_next}
      do
        for lambda_c_all in ${lambdas_c_all}
        do
          for q_all_loss in ${qs_all_loss}
          do
            for c_all_loss in ${cs_all_loss}
            do
              for c_next_loss in ${cs_next_loss}
              do
                for num_mlp_layer in "${nums_mlp_layer[@]}"
                do
                  for dropout in ${dropouts}
                  do
echo -e "emb: 64, weight_decay: ${weight_decay}, lambda_q_all: ${lambda_q_all}, lambda_c_next: ${lambda_c_next}, lambda_c_all: ${lambda_c_all}, q_all_loss: ${q_all_loss}, c_all_loss: ${c_all_loss}, c_next_loss: ${c_next_loss}, num_mlp_layer: ${num_mlp_layer}, dropout: ${dropout}"
                    python /ghome/xiongzj/code/dlkt/example/train/qikt.py \
                      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                      --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                      --train_strategy "valid_test" --num_epoch 200 \
                      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                      --main_metric "AUC" --use_multi_metrics False \
                      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                      --train_batch_size 32 --evaluate_batch_size 128 \
                      --enable_clip_grad False --grad_clipped 10.0 \
                      --num_concept 246 --num_question 5730 \
                      --dim_emb 64 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" --num_mlp_layer "${num_mlp_layer}" \
                      --lambda_q_all "${lambda_q_all}" --lambda_c_next "${lambda_c_next}" --lambda_c_all "${lambda_c_all}" --use_irt True \
                      --weight_predict_q_all_loss "${q_all_loss}" --weight_predict_q_next_loss 1 \
                      --weight_predict_c_all_loss "${c_all_loss}" --weight_predict_c_next_loss "${c_next_loss}" \
                      --save_model False --debug_mode False --seed 0

echo -e "emb: 256, weight_decay: ${weight_decay}, lambda_q_all: ${lambda_q_all}, lambda_c_next: ${lambda_c_next}, lambda_c_all: ${lambda_c_all}, q_all_loss: ${q_all_loss}, c_all_loss: ${c_all_loss}, c_next_loss: ${c_next_loss}, num_mlp_layer: ${num_mlp_layer}, dropout: ${dropout}"
                    python /ghome/xiongzj/code/dlkt/example/train/qikt.py \
                      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                      --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                      --train_strategy "valid_test" --num_epoch 200 \
                      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                      --main_metric "AUC" --use_multi_metrics False \
                      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                      --train_batch_size 32 --evaluate_batch_size 128 \
                      --enable_clip_grad False --grad_clipped 10.0 \
                      --num_concept 246 --num_question 5730 \
                      --dim_emb 256 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" --num_mlp_layer "${num_mlp_layer}" \
                      --lambda_q_all "${lambda_q_all}" --lambda_c_next "${lambda_c_next}" --lambda_c_all "${lambda_c_all}" --use_irt True \
                      --weight_predict_q_all_loss "${q_all_loss}" --weight_predict_q_next_loss 1 \
                      --weight_predict_c_all_loss "${c_all_loss}" --weight_predict_c_next_loss "${c_next_loss}" \
                      --save_model False --debug_mode False --seed 0
                  done
                done
              done
            done
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qikt_our_setting_slepemapy_fold_0_ob.txt