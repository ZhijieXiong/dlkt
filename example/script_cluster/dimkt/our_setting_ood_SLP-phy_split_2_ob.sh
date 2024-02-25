#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="SLP-phy"
  data_type="single_concept"
  split=2


  dims_emb=(64 128)
  nums_min_question=(10 15)
  dropouts='0.1 0.2 0.3'
  weights_decay='0.0001 0.00001'
  nums_diff=(50 100)
  for weight_decay in ${weights_decay}
  do
    for dim_emb in "${dims_emb[@]}"
    do
      for num_min_question in "${nums_min_question[@]}"
      do
        for num_diff in "${nums_diff[@]}"
        do
          for dropout in ${dropouts}
          do
            echo -e "weights_decay: ${weight_decay}, dim_emb: ${dim_emb}, num_min_question: ${num_min_question}, num_diff: ${num_diff}, dropout: ${dropout}"
            python /ghome/xiongzj/code/dlkt/example/train/dimkt.py \
              --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
              --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
              --train_strategy valid_test --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 \
              --use_last_average False --epoch_last_average 5 \
              --main_metric "AUC" --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 5 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
              --train_batch_size 64 --evaluate_batch_size 256 \
              --enable_clip_grad False --grad_clipped 10.0 \
              --num_concept 54 --num_question 1915 \
              --dim_emb "${dim_emb}" --dropout "${dropout}" \
              --num_min_question "${num_min_question}" --num_min_concept 30 --num_question_diff "${num_diff}" --num_concept_diff "${num_diff}" \
              --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
              --save_model False --debug_mode False --use_cpu False --seed 0

            rm /ghome/xiongzj/code/dlkt/lab/settings/our_setting_ood_by_school/SLP-phy_train_split_2_dimkt_diff.json
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dimkt_our_setting_ood_SLP-phy_split_2_ob.txt