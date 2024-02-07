#!/usr/bin/env bash

{
  dataset_name="assist2012"
  data_type="single_concept"
  fold=0

  weights_adv_pred_loss='1'
  etas='5 10 20'
  gammas='5 10 20'
  dropouts='0.2'
  adv_learning_rates='1 5 10'
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
            echo -e "lr: 0.0001, no lr decay, adv_learning_rate: ${adv_learning_rate}, dropout: ${dropout}, weight_adv_pred_loss: ${weight_adv_pred_loss}, eta: ${eta}, gamma: ${gamma}"
            python /ghome/xiongzj/code/dlkt/example/train/akt_max_entropy_aug.py \
              --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
              --optimizer_type adam --weight_decay 0 --momentum 0.9 \
              --train_strategy valid_test --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric AUC --use_multi_metrics False \
              --learning_rate 0.0004 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
              --train_batch_size 64 --evaluate_batch_size 128 \
              --enable_clip_grad True --grad_clipped 10.0 \
              --num_concept 265 --num_question 53091 \
              --dim_model 256 --key_query_same True --num_head 8 --num_block 3 --dim_ff 256 --dim_final_fc 512 --dropout "${dropout}" \
              --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
              --use_warm_up False --epoch_warm_up 4 \
              --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss "${weight_adv_pred_loss}" --loop_adv 3 --adv_learning_rate "${adv_learning_rate}" --eta "${eta}" --gamma "${gamma}" \
              --save_model False --debug_mode False --seed 0
          done
        done
      done
    done
  done

} >> /ghome/xiongzj/code/dlkt/example/result_cluster/akt_ME-ADA_our_setting_assist2012_fold_0_ob.txt