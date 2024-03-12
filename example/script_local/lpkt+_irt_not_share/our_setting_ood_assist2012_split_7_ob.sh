#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="assist2012"
  data_type="single_concept"
  split=7

  weights_decay='0.00001 0.000001'
  dims=(64 128)
  dropouts='0.1 0.2'
  for weight_decay in ${weights_decay}
  do
    for dim in "${dims[@]}"
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, dim: ${dim}, dropout: ${dropout}"
        python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
          --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 20 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 128 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 265 --num_question 53091 --ablation_set 0 \
          --dim_question "${dim}" --dim_latent "${dim}" --dim_correct 50 --dropout "${dropout}" \
          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 \
          --save_model False --debug_mode False --use_cpu False --seed 0
      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/lpkt_plus_v2_our_setting_ood_assist2012_split_7_ob.txt