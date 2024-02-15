#!/usr/bin/env bash

{
  dataset_name="bridge2algebra2006"
  data_type="single_concept"
  fold=0

  dropouts='0.1 0.2'
  weight_decays='0 0.0001 0.00001'
  weights_QT_loss='0.3 0.7 1'
  weights_IK_loss='0.7'
  for weight_decay in ${weight_decays}
  do
    for weight_QT_loss in ${weights_QT_loss}
      do
        for weight_IK_loss in ${weights_IK_loss}
        do
          for dropout in ${dropouts}
          do
            echo -e "weight_decay: ${weight_decay}, weight_QT_loss: ${weight_QT_loss}, weight_IK_loss: ${weight_IK_loss}, dropout: ${dropout}"
            python /ghome/xiongzj/code/dlkt/example/train/at_dkt.py \
              --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
              --optimizer_type adam --weight_decay "${weight_decay}" --momentum 0.9 \
              --train_strategy valid_test --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric AUC --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
              --train_batch_size 64 --evaluate_batch_size 256 \
              --enable_clip_grad False --grad_clipped 10.0 \
              --num_concept 493 --num_question 129263 \
              --dim_emb 256 --dim_latent 256 --rnn_type "lstm" --num_rnn_layer 1 --dropout "${dropout}" \
              --QT_net_type "transformer" --QT_transformer_num_block 4 --QT_rnn_type "lstm" --QT_num_rnn_layer 4 \
              --QT_transformer_num_head 4 --IK_start 10 --weight_QT_loss "${weight_QT_loss}" --weight_IK_loss "${weight_IK_loss}" \
              --save_model False --debug_mode False --seed 0
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/at_dkt_our_setting_bridge2algebra2006_fold_0_ob.txt