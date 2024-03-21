#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/qdkt_max_entropy_aug.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 123 --num_question 17751 \
      --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.4 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
      --use_warm_up False --epoch_warm_up 4 \
      --epoch_interval_generate 1 --epoch_generate 200 --weight_adv_pred_loss 1 --loop_adv 3 --adv_learning_rate 10 --eta 5 --gamma 10 \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done

} >> F:/code/myProjects/dlkt/example/result_local/qdkt_ME-ADA_our_setting_assist2009_save.txt