#!/usr/bin/env bash

{
  dataset_name="slepemapy"
  data_type="single_concept"
  fold=0

  dropouts='0.1 0.2 0.3'
  weights_decay='0.0001 0.00001'
  for weight_decay in ${weights_decay}
  do
    for num_min_question in 100 150
    do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, num_min_question: ${num_min_question}, dropout: ${dropout}"
        python F:/code/myProjects/dlkt/example/train/dimkt.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy valid_test --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 \
          --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 5 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_min_question "${num_min_question}" --num_min_concept 30 --num_concept 246 --num_question 5730 \
          --dim_emb 128 --num_question_diff 100 --num_concept_diff 100 --dropout "${dropout}" \
          --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
          --save_model False --seed 0

        rm F:/code/myProjects/dlkt/lab/settings/our_setting/slepemapy_train_fold_0_dimkt_diff.json

      done
    done
  done

} >> F:/code/myProjects/dlkt/example/result_local/dimkt_our_setting_slepemapy_fold_0_ob.txt