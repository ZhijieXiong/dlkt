#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="SLP-his"
  data_type="single_concept"
  split=6

  weights_decay='0.001 0.0001 0.00001 0.000001 0'
  weights_enhance_loss1='0.01 0.1 1'
  weights_enhance_loss2='0.01 0.1 1'
  dropouts='0.2'
  for weight_decay in ${weights_decay}
  do
    for weight_enhance_loss1 in ${weights_enhance_loss1}
    do
      for weight_enhance_loss2 in ${weights_enhance_loss2}
      do
        for dropout in ${dropouts}
        do
          echo -e "weight_decay: ${weight_decay}, weight_enhance_loss1: ${weight_enhance_loss1}, weight_enhance_loss2: ${weight_enhance_loss2}, dropout: ${dropout}"
          python /ghome/xiongzj/code/dlkt/example/train/qdkt_output_enhance.py \
            --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
            --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
            --train_strategy "valid_test" --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_concept 22 --num_question 1251 \
            --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
            --enhance_method 0 \
            --weight_enhance_loss1 "${weight_enhance_loss1}" --num_min_question4diff 15 --hard_acc 0.55 --easy_acc 0.85 \
            --weight_enhance_loss2 "${weight_enhance_loss2}" --enhance_method2_update_few_shot True --num_few_shot 3 \
            --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb False \
            --save_model False --seed 0
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_output_enhance_our_setting_SLP-his_split_6_ob.txt