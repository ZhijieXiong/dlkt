#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_school"
  dataset_name="SLP-his"
  data_type="single_concept"
  split=6


  dims_model=(256 512)
  nums_block=(1 2)
  nums_head=(4 8)
  nums_min_question=(10 15)
  dropouts='0.1 0.2 0.3'
  weights_decay='0.0001 0.00001'

  for dim_model in "${dims_model[@]}"
  do
    for num_block in "${nums_block[@]}"
    do
      for num_head in "${nums_head[@]}"
      do
        for dropout in ${dropouts}
        do
          echo -e "dim_model: ${dim_model}, num_block: ${num_block}, num_head: ${num_head}, dropout: ${dropout}"
          python /ghome/xiongzj/code/dlkt/example/train/dimkt.py \
            --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
            --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
            --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
            --train_strategy "valid_test" --num_epoch 200 \
            --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
            --main_metric "AUC" --use_multi_metrics False \
            --learning_rate 0.0001 --enable_lr_schedule False \
            --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
            --train_batch_size 24 --evaluate_batch_size 128 \
            --enable_clip_grad True --grad_clipped 10.0 \
            --num_concept 22 --num_question 1251 \
            --dim_model "${dim_model}" --key_query_same True --num_head "${num_head}" \
            --num_block "${num_block}" --dim_ff "${dim_model}" --dim_final_fc 512 --dropout "${dropout}" \
            --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
            --save_model False --debug_mode False --seed 0
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/dimkt_our_setting_ood_SLP-his_split_6_ob.txt