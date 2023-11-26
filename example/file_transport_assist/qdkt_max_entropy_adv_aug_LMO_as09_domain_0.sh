#!/bin/bash

{
  seeds=(0 1 2 3 4)
  dataset_name="assist2009"
  domain=0

  for seed in "${seeds[@]}"
  do
    echo -e "seed: ${seed}"
    python /ghome/xiongzj/code/dlkt/example/train/qdkt_max_entropy_aug.py \
      --setting_name "random_split_leave_multi_out_setting" --data_type "multi_concept" \
      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
      --learning_rate 0.0005 --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --enable_clip_grad False --grad_clipped 10 \
      --num_concept 123 --num_question 17751 --dim_concept 64 --dim_question 64 --dim_correct 128 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
      --use_warm_up False --epoch_warm_up 4 --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 100 \
      --weight_adv_pred_loss 10 --adv_learning_rate 10 --eta 20 --gamma 10 \
      --save_model False --seed "${seed}"

  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/baseline/save/qdkt_max_entropy_adv_aug_LMO_as09_domain_0.txt