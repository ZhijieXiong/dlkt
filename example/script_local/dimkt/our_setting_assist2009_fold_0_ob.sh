#!/usr/bin/env bash

#{
  dataset_name="assist2009"
  data_type="only_question"
  fold=0

  dropouts='0.1 0.2 0.3'
  weight_decays='0.001 0.0001'
  for weight_decay in ${weight_decays}
  do
    for dim_emb in 64 128
      do
        for num_min_question in 10 15 20
        do
          for num_diff in 50 75 100
          do
            for dropout in ${dropouts}
            do
              echo -e "weight_decay: ${weight_decay}, dim_emb: ${dim_emb}, num_min_question: ${num_min_question}, num_diff: ${num_diff}, dropout: ${dropout}"
              python F:/code/myProjects/dlkt/example/train/dimkt.py \
                --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                --train_strategy valid_test --num_epoch 200 \
                --use_early_stop True --epoch_early_stop 10 \
                --use_last_average False --epoch_last_average 5 \
                --main_metric "AUC" --use_multi_metrics False \
                --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                --train_batch_size 64 --evaluate_batch_size 256 \
                --enable_clip_grad False --grad_clipped 10.0 \
                --num_min_question "${num_min_question}" --num_min_concept 30 --num_concept 123 --num_question 17751 \
                --dim_emb "${dim_emb}" --num_question_diff "${num_diff}" --num_concept_diff "${num_diff}" --dropout "${dropout}" \
                --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
                --transfer_head2zero False --head2tail_transfer_method "mean_pool" \
                --save_model False --seed 0
          done
        done
      done
    done
  done
#} >> F:/code/myProjects/dlkt/example/result_local/dimktkt_our_setting_assist2009_fold_0_ob.txt