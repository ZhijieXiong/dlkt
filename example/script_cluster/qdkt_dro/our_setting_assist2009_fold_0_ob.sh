#!/usr/bin/env bash

#{
  setting_name="our_setting_ood_by_school"
  dataset_name="SLP-phy"
  data_type="single_concept"
  split=2

  batch_sizes=(4096 2048 1024 512)
  weights_decay='0.0001 0.00001 0.000001 0'
  dropouts='0.1 0.2 0.3 0.4 0.5'
  for batch_size in "${batch_sizes[@]}"
  do
    for weight_decay in ${weights_decay}
    do
      for dropout in ${dropouts}
      do
        echo -e "use dro: False, batch_size: ${batch_size}, weight_decay: ${weight_decay}, dropout: ${dropout}"
        python /Users/dream/Desktop/code/projects/dlkt/example/train/qdkt_dro.py \
          --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_${split}.txt" --test_file_name "${dataset_name}_test_ood_${split}.txt" \
          --optimizer_type "adam" --weight_decay "${dropout}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size "${batch_size}" --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 54 --num_question 1915 \
          --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
          --use_dro False --beta 5.0 --alpha 0.001 --max_seq_len 200 \
          --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
          --save_model False --debug_mode False --seed 0
      done
    done
  done

  batch_sizes=(1024)
  weights_decay='0'
  dropouts='0.1'
  betas='0.1 1 10'
  alphas='0.001 0.01 0.1 1'
  for batch_size in "${batch_sizes[@]}"
  do
    for dropout in ${dropouts}
    do
      for weight_decay in ${weights_decay}
      do
        for beta in ${betas}
        do
          for alpha in ${alphas}
          do
            echo -e "use dro: True, weight_decay: ${weight_decay}, dropout: ${dropout}, beta: ${beta}, alpha: ${alpha}"
            python /Users/dream/Desktop/code/projects/dlkt/example/train/qdkt_dro.py \
              --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_${split}.txt" --test_file_name "${dataset_name}_test_ood_${split}.txt" \
              --optimizer_type "adam" --weight_decay "${dropout}" --momentum 0.9 \
              --train_strategy "valid_test" --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric "AUC" --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
              --train_batch_size "${batch_size}" --evaluate_batch_size 256 \
              --enable_clip_grad False --grad_clipped 10.0 \
              --num_concept 54 --num_question 1915 \
              --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
              --use_dro True --beta "${beta}" --alpha "${alpha}" --max_seq_len 200 \
              --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
              --save_model False --debug_mode False --seed 0
          done
        done
      done
    done
  done

#} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_dro_our_setting_ood_by_school_SLP-phy_split_2_ob1.txt

