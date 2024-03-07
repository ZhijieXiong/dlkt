#!/usr/bin/env bash

{
  dataset_name="slepemapy"
  data_type="single_concept"
  fold=0


  weight_decays='0.0001 0.00001 0.000001'
  dropouts='0.05 0.1 0.15'
  dims=(64 128)
  for weight_decay in ${weight_decays}
  do
    for dim in "${dims[@]}"
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, dim: ${dim}, dropout: ${dropout}"
        python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 20 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 1024 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 246 --num_question 5730 --ablation_set 0 \
          --dim_question "${dim}" --dim_latent "${dim}" --dim_correct 50 --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/lpkt+_baseline_our_setting_slepemapy_fold_0_ob.txt