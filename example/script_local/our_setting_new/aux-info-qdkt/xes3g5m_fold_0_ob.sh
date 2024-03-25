#!/usr/bin/env bash

{
  dataset_name="xes3g5m"
  data_type="only_question"
  fold=0

  dropouts='0.1'
  weights_decay='0.0001 0.00001 0.000001 0'
  dims_question=(64 128)
  dims_latent=(64 128)
  nums_predict_layer=(2 3)
  weights_aux_emb='0.5'
  # 1
  nums_rnn_layer=(2 3)
  for weight_decay in ${weights_decay}
  do
    for weight_aux_emb in ${weights_aux_emb}
    do
      for dim_question in "${dims_question[@]}"
      do
        for dim_latent in "${dims_latent[@]}"
        do
          for num_rnn_layer in "${nums_rnn_layer[@]}"
          do
            for num_predict_layer in "${nums_predict_layer[@]}"
            do
              for dropout in ${dropouts}
              do
                echo -e "weight_decay: ${weight_decay}, weight_aux_emb: ${weight_aux_emb}, dim_question: ${dim_question}, dim_latent: ${dim_latent}, num_rnn_layer: ${num_rnn_layer}, num_predict_layer: ${num_predict_layer}, dropout: ${dropout}"
                python F:/code/myProjects/dlkt/example/train/aux_info_qdkt.py \
                  --setting_name "our_setting_new" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                  --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                  --optimizer_type adam --weight_decay "${weight_decay}" --momentum 0.9 \
                  --train_strategy valid_test --num_epoch 200 \
                  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                  --main_metric AUC --use_multi_metrics False \
                  --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                  --train_batch_size 64 --evaluate_batch_size 256 \
                  --enable_clip_grad False --grad_clipped 10.0 \
                  --num_concept 865 --num_question 7652 \
                  --weight_aux_emb "${weight_aux_emb}" --dim_question "${dim_question}" --dim_latent "${dim_latent}" --rnn_type "gru" --num_rnn_layer "${num_rnn_layer}" --dropout "${dropout}" \
                  --num_predict_layer "${num_predict_layer}" --dim_predict_mid 128 --activate_type "relu" \
                  --save_model False --debug_mode False --use_cpu False --seed 0
              done
            done
          done
        done
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/aux-info-qdkt_our_setting_new_xes3g5m_fold_0_ob.txt