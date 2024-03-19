#!/usr/bin/env bash

dataset_name="ednet-kt1"
data_type="only_question"
fold=0
num_rnn_layer=1
num_mlp_layer=2
dim_question=64
dim_latent=128

dropouts='0.1'
weight_decays='0.0001 0.00001 0'

ws_penalty_neg='0.0001 0.001 0.01 0.1'
ws_learning='0.0001 0.001 0.01 0.1'
ws_counter_fact='0.0001 0.001 0.01 0.1'
ws_q_able='0.0001 0.001 0.01 0.1'

{
  echo -e "DCT IRT: add single aux penalty_neg loss"
  for weight_decay in ${weight_decays}
  do
    for w_penalty_neg in ${ws_penalty_neg}
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, w_penalty_neg: ${w_penalty_neg}, dropout: ${dropout}"
        python /ghome/xiongzj/code/dlkt/example/train/dct.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 15 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 188 --num_question 11858 \
          --multi_stage False --test_theory "irt" \
          --dim_question "${dim_question}" --dim_correct "${dim_question}" --dim_latent "${dim_latent}" --rnn_type "gru" \
          --num_rnn_layer "${num_rnn_layer}" --num_mlp_layer "${num_mlp_layer}" --que_user_share_proj False --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 \
          --w_penalty_neg "${w_penalty_neg}" --w_learning 0 --w_counter_fact 0 --w_q_table 0 \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dct_new_irt_not_share_single_stage_our_setting_ednet-kt1_fold_0_ob1.txt

{
  echo -e "DCT IRT: add single aux learn loss"
  for weight_decay in ${weight_decays}
  do
    for w_learning in ${ws_learning}
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, w_learning: ${w_learning}, dropout: ${dropout}"
        python /ghome/xiongzj/code/dlkt/example/train/dct.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 15 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 188 --num_question 11858 \
          --multi_stage False --test_theory "irt" \
          --dim_question "${dim_question}" --dim_correct "${dim_question}" --dim_latent "${dim_latent}" --rnn_type "gru" \
          --num_rnn_layer "${num_rnn_layer}" --num_mlp_layer "${num_mlp_layer}" --que_user_share_proj False --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 \
          --w_penalty_neg 0 --w_learning "${w_learning}" --w_counter_fact 0 --w_q_table 0 \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dct_new_irt_not_share_single_stage_our_setting_ednet-kt1_fold_0_ob2.txt


{
  echo -e "DCT IRT: add single aux counter_fact loss"
  for weight_decay in ${weight_decays}
  do
    for w_counter_fact in ${ws_counter_fact}
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, w_counter_fact: ${w_counter_fact}, dropout: ${dropout}"
        python /ghome/xiongzj/code/dlkt/example/train/dct.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 15 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 188 --num_question 11858 \
          --multi_stage False --test_theory "irt" \
          --dim_question "${dim_question}" --dim_correct "${dim_question}" --dim_latent "${dim_latent}" --rnn_type "gru" \
          --num_rnn_layer "${num_rnn_layer}" --num_mlp_layer "${num_mlp_layer}" --que_user_share_proj False --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 \
          --w_penalty_neg 0 --w_learning 0 --w_counter_fact "${w_counter_fact}" --w_q_table 0 \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dct_new_irt_not_share_single_stage_our_setting_ednet-kt1_fold_0_ob3.txt


{
  echo -e "DCT IRT: add single aux q_table loss"
  for weight_decay in ${weight_decays}
  do
    for w_q_able in ${ws_q_able}
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, w_q_able: ${w_q_able}, dropout: ${dropout}"
        python /ghome/xiongzj/code/dlkt/example/train/dct.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 15 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 188 --num_question 11858 \
          --multi_stage False --test_theory "irt" \
          --dim_question "${dim_question}" --dim_correct "${dim_question}" --dim_latent "${dim_latent}" --rnn_type "gru" \
          --num_rnn_layer "${num_rnn_layer}" --num_mlp_layer "${num_mlp_layer}" --que_user_share_proj False --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 \
          --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 --w_q_table "${w_q_able}" \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dct_new_irt_not_share_single_stage_our_setting_ednet-kt1_fold_0_ob4.txt
