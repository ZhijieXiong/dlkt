#!/usr/bin/env bash

{
  dataset_name="edi2020-task1"
  data_type="single_concept"
  fold=0

  dropouts='0.2'
  weights_decay='0.00001 0.000001'
  for weight_decay in ${weights_decay}
  do
      for dropout in ${dropouts}
      do
        echo -e "weight_decay: ${weight_decay}, dropout: ${dropout}"
        python /ghome/xiongzj/code/dlkt/example/train/dimkt.py \
          --setting_name "dimkt_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 \
          --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 5 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 282 --num_question 27613 --num_min_question 30 --num_min_concept 30 \
          --dim_emb 128 --num_question_diff 100 --num_concept_diff 100 --dropout "${dropout}" \
          --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
          --save_model False --debug_mode False --use_cpu False --seed 0
    done
  done

} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dimkt_dimkt_setting_edi2020-task1_fold_0_ob.txt