#!/usr/bin/env bash

{
  dataset_name="assist2017"
  data_type="single_concept"
  fold=0



  echo -e "weight_decay: 0.0001, dim_emb: 64, num_min_question: 100, num_min_concept: 50, num_diff: 100, dropout: 0.3"
  python F:/code/myProjects/dlkt/example/train/dimkt.py \
    --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
    --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
    --train_strategy valid_test --num_epoch 200 \
    --use_early_stop True --epoch_early_stop 10 \
    --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False \
    --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
    --train_batch_size 64 --evaluate_batch_size 256 \
    --enable_clip_grad False --grad_clipped 10.0 \
    --num_min_question 100 --num_min_concept 50 --num_concept 101 --num_question 2803 \
    --dim_emb 64 --num_question_diff 100 --num_concept_diff 100 --dropout 0.3 \
    --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
    --transfer_head2zero False --head2tail_transfer_method "mean_pool" \
    --save_model False --seed 0
  rm F:/code/myProjects/dlkt/lab/settings/our_setting/assist2017_train_fold_0_dimkt_diff.json



  dropouts='0.1 0.2 0.3'
  for num_min_question in 150 200
  do
    for num_min_concept in 30 50
    do
      for num_diff in 50 100
      do
        for dropout in ${dropouts}
        do
          echo -e "weight_decay: 0.0001, dim_emb: 64, num_min_question: ${num_min_question}, num_min_concept: ${num_min_concept}, num_diff: ${num_diff}, dropout: ${dropout}"
          python F:/code/myProjects/dlkt/example/train/dimkt.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
            --train_strategy valid_test --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 \
            --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_min_question "${num_min_question}" --num_min_concept "${num_min_concept}" --num_concept 101 --num_question 2803 \
            --dim_emb 64 --num_question_diff "${num_diff}" --num_concept_diff "${num_diff}" --dropout "${dropout}" \
            --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
            --transfer_head2zero False --head2tail_transfer_method "mean_pool" \
            --save_model False --seed 0

          rm F:/code/myProjects/dlkt/lab/settings/our_setting/assist2017_train_fold_0_dimkt_diff.json
        done
      done
    done
  done






  dropouts='0.1 0.2 0.3'
  for num_min_question in 100 150 200
  do
    for num_min_concept in 30 50
    do
      for num_diff in 50 100
      do
        for dropout in ${dropouts}
        do
          echo -e "weight_decay: 0.0001, dim_emb: 128, num_min_question: ${num_min_question}, num_min_concept: ${num_min_concept}, num_diff: ${num_diff}, dropout: ${dropout}"
          python F:/code/myProjects/dlkt/example/train/dimkt.py \
            --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
            --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
            --train_strategy valid_test --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 \
            --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
            --train_batch_size 64 --evaluate_batch_size 256 \
            --enable_clip_grad False --grad_clipped 10.0 \
            --num_min_question "${num_min_question}" --num_min_concept "${num_min_concept}" --num_concept 101 --num_question 2803 \
            --dim_emb 128 --num_question_diff "${num_diff}" --num_concept_diff "${num_diff}" --dropout "${dropout}" \
            --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
            --transfer_head2zero False --head2tail_transfer_method "mean_pool" \
            --save_model False --seed 0

          rm F:/code/myProjects/dlkt/lab/settings/our_setting/assist2017_train_fold_0_dimkt_diff.json
        done
      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/dimkt_our_setting_assist2017_fold_0_ob.txt