#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"
  fold=0

#  dct_ME-ADA_our_setting_assist2009_fold_0_ob1.txt
#  weights_decay='0.0001 0.00001'
#  nums_rnn_layer=(1 2)
#  dims_question=(32 64)
#  dims_latent=(32 64)
#  dropouts='0.1'
#  weights_adv_pred_loss='1'
#  adv_learning_rates='1 10'
#  etas='5 10 20'
#  gammas='5 10 20'


# dct_ME-ADA_our_setting_assist2009_fold_0_ob2.txt
  weights_decay='0.001'
  nums_rnn_layer=(2)
  dims_question=(32 64)
  dims_latent=(32 64)
  dropouts='0.1'
  weights_adv_pred_loss='1'
  adv_learning_rates='1 10'
  etas='5 10 20'
  gammas='5 10 20'
  for weight_decay in ${weights_decay}
  do
    for weight_adv_pred_loss in ${weights_adv_pred_loss}
    do
      for eta in ${etas}
      do
        for gamma in ${gammas}
        do
          for dropout in ${dropouts}
          do
            for adv_learning_rate in ${adv_learning_rates}
            do
              for num_rnn_layer in "${nums_rnn_layer[@]}"
              do
                for dim_question in "${dims_question[@]}"
                do
                  for dim_latent in "${dims_latent[@]}"
                  do
                    echo -e "weight_decay: ${weight_decay}, num_rnn_layer: ${num_rnn_layer}, dim_question: ${dim_question}, dim_latent: ${dim_latent}, dropout: ${dropout}"
                    echo -e "adv_learning_rate: ${adv_learning_rate}, weight_adv_pred_loss: ${weight_adv_pred_loss}, eta: ${eta}, gamma: ${gamma}"
                    python F:/code/myProjects/dlkt/example/train/dct_max_entropy_aug.py \
                      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                      --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                      --train_strategy "valid_test" --num_epoch 200 \
                      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                      --main_metric "AUC" --use_multi_metrics False \
                      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                      --train_batch_size 64 --evaluate_batch_size 256 \
                      --enable_clip_grad False --grad_clipped 10.0 \
                      --num_concept 123 --num_question 17751 \
                      --dim_question "${dim_question}" --dim_correct 64 --dim_latent "${dim_latent}" --rnn_type "gru" --num_rnn_layer "${num_rnn_layer}" --dropout "${dropout}" \
                      --use_warm_up False --epoch_warm_up 4 \
                      --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss "${weight_adv_pred_loss}" --loop_adv 3 --adv_learning_rate "${adv_learning_rate}" --eta "${eta}" --gamma "${gamma}" \
                      --save_model False --debug_mode False --use_cpu False --seed 0
                  done
                done
              done
            done
          done
        done
      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/dct_ME-ADA_our_setting_assist2009_fold_0_ob2.txt