#!/usr/bin/env bash

#{
  dataset_name="assist2009"
  data_type="only_question"
  fold=0

  weights_question_loss='0.01 0.1 1'
  gammas4transfer_question='0 0.1 1 10'
  min_context_seq_lens=(10)
  head_seq_lens=(20)
  for min_context_seq_len in "${min_context_seq_lens[@]}"
  do
    for head_seq_len in "${head_seq_lens[@]}"
    do
      for gamma4transfer_question in ${gammas4transfer_question}
      do
        for weight_question_loss in ${weights_question_loss}
        do
          echo -e "two_branch4question_transfer: True, min_context_seq_len: ${min_context_seq_len}, head_seq_len: ${head_seq_len}, gamma4transfer_question: ${gamma4transfer_question}, weight_question_loss: ${weight_question_loss}"
          python /ghome/xiongzj/code/dlkt/example/train/qdkt_matual_enhance4long_tail.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type "adam" --weight_decay 0.001 --momentum 0.9 \
            --train_strategy "valid_test" --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 \
            --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_concept 123 --num_question 17751 \
            --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
            --two_stage False --max_seq_len 200 \
            --min_context_seq_len "${min_context_seq_len}" --head_question_threshold 0.8 --head_seq_len "${head_seq_len}" --use_transfer4seq False \
            --beta4transfer_seq 1 --gamma4transfer_question "${gamma4transfer_question}" --only_update_low_fre True --two_branch4question_transfer True \
            --save_model_dir "" --optimizer_type_question_branch "adam" --weight_decay_question_branch 0 --momentum_question_branch 0.9 \
            --learning_rate_question_branch 0.0001 --enable_lr_schedule_question_branch False --lr_schedule_type_question_branch "MultiStepLR" \
            --lr_schedule_step_question_branch 10 --lr_schedule_milestones_question_branch "[5]" --lr_schedule_gamma_question_branch 0.5 \
            --enable_clip_grad_question_branch True --grad_clipped_question_branch 5.0 \
            --weight_seq_loss 0.1 --weight_question_loss "${weight_question_loss}" \
            --save_model False --seed 0

          echo -e "two_branch4question_transfer: False, min_context_seq_len: ${min_context_seq_len}, head_seq_len: ${head_seq_len}, gamma4transfer_question: ${gamma4transfer_question}, weight_question_loss: ${weight_question_loss}"
          python /ghome/xiongzj/code/dlkt/example/train/qdkt_matual_enhance4long_tail.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type "adam" --weight_decay 0.001 --momentum 0.9 \
            --train_strategy "valid_test" --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 \
            --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_concept 123 --num_question 17751 \
            --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
            --two_stage False --max_seq_len 200 \
            --min_context_seq_len "${min_context_seq_len}" --head_question_threshold 0.8 --head_seq_len "${head_seq_len}" --use_transfer4seq False \
            --beta4transfer_seq 1 --gamma4transfer_question "${gamma4transfer_question}" --only_update_low_fre True --two_branch4question_transfer False \
            --save_model_dir "" --optimizer_type_question_branch "adam" --weight_decay_question_branch 0 --momentum_question_branch 0.9 \
            --learning_rate_question_branch 0.0001 --enable_lr_schedule_question_branch False --lr_schedule_type_question_branch "MultiStepLR" \
            --lr_schedule_step_question_branch 10 --lr_schedule_milestones_question_branch "[5]" --lr_schedule_gamma_question_branch 0.5 \
            --enable_clip_grad_question_branch True --grad_clipped_question_branch 5.0 \
            --weight_seq_loss 0.1 --weight_question_loss "${weight_question_loss}" \
            --save_model False --seed 0
        done
      done
    done
  done

#} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_MELT_one_stage_not_use_seq_branch_our_setting_assist2009_fold_0_ob.txt
