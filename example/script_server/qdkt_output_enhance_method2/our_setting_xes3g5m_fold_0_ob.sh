#!/usr/bin/env bash

{
  dataset_name="xes3g5m"
  data_type="only_question"
  fold=0

  weights_decay='0.0001 0.00001 0'
  weights_enhance_loss2='0.0001 0.001 0.01 0.1 1 3'
  for weight_decay in ${weights_decay}
  do
    for weight_enhance_loss2 in ${weights_enhance_loss2}
    do
      echo -e "weight_decay: ${weight_decay}, weight_enhance_loss2: ${weight_enhance_loss2}"
      CUDA_VISIBLE_DEVICES=2 python /home/xiongzj/myProjects/KT/dlkt/example/train/qdkt_output_enhance.py \
        --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
        --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
        --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
        --train_strategy "valid_test" --num_epoch 200 \
        --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
        --main_metric "AUC" --use_multi_metrics False \
        --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
        --train_batch_size 64 --evaluate_batch_size 256 \
        --enable_clip_grad False --grad_clipped 10.0 \
        --num_concept 865 --num_question 7652 \
        --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
        --enhance_method 2 \
        --weight_enhance_loss1 1 --num_min_question4diff 100 --hard_acc 0.4 --easy_acc 0.8 \
        --weight_enhance_loss2 "${weight_enhance_loss2}" --enhance_method2_update_few_shot True --num_few_shot 5 \
        --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
        --save_model False --seed 0
    done
  done
} >> /home/xiongzj/myProjects/KT/dlkt/example/results/qdkt_output_enhance_method2_our_setting_xes3g5m_fold_0_ob.txt