#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_country"
  dataset_name="slepemapy"
  data_type="single_concept"
  split=0


  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 20 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 246 --num_question 5730 --ablation_set 0 \
      --dim_question 128 --dim_latent 128 --dim_correct 50 --dropout 0.2 --que_user_share_proj False \
      --multi_stage True --test_theory "irt" \
      --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 --w_q_table 0 \
      --save_model True --debug_mode False --use_cpu False --seed "${seed}"
  done

} >> F:/code/myProjects/dlkt/example/result_local/lpkt+_irt_not_share_baseline_our_setting_ood_slepemapy_split_0_save_ob.txt