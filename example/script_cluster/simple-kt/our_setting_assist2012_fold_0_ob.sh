#!/usr/bin/env bash

{
  dataset_name="assist2012"
  data_type="single_concept"
  fold=0

  dropouts='0.1 0.2 0.3'
  weight_decays='0'
  dims_model=(256)
  nums_block=(2 4)
  nums_head=(4 8)
  dims_ff=(256)
  dims_final_fc=(256 512)
  dims_final_fc2=(256 512)
  for weight_decay in ${weight_decays}
  do
    for dim_model in "${dims_model[@]}"
    do
      for num_block in "${nums_block[@]}"
      do
        for num_head in "${nums_head[@]}"
        do
          for dim_ff in "${dims_ff[@]}"
          do
            for dim_final_fc in "${dims_final_fc[@]}"
            do
              for dim_final_fc2 in "${dims_final_fc2[@]}"
              do
                  for dropout in ${dropouts}
                  do
echo -e "weight_decay: ${weight_decay}, dim_model: ${dim_model}, num_block: ${num_block}, num_head: ${num_head}, dim_ff: ${dim_ff}, dim_final_fc: ${dim_final_fc}, dim_final_fc2: ${dim_final_fc2}, dropout: ${dropout}"
                    python /ghome/xiongzj/code/dlkt/example/train/simple_kt.py \
                      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                      --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                      --train_strategy "valid_test" --num_epoch 200 \
                      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                      --main_metric "AUC" --use_multi_metrics False \
                      --learning_rate 0.0004 --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
                      --train_batch_size 64 --evaluate_batch_size 128 \
                      --enable_clip_grad False --grad_clipped 10.0 \
                      --num_concept 265 --num_question 53091 \
                      --dim_model "${dim_model}" --num_block "${num_block}" --num_head "${num_head}" --dim_ff "${dim_ff}" \
                      --dim_final_fc "${dim_final_fc}" --dim_final_fc2 "${dim_final_fc2}" --dropout "${dropout}" \
                      --seq_len 200 --key_query_same True --separate_qa False --difficulty_scalar False \
                      --use_sample_weight False --sample_weight_method "highlight_tail" --tail_weight 1.1 \
                      --use_LLM_emb4question False --use_LLM_emb4concept False --train_LLM_emb True \
                      --save_model False --debug_mode False --use_cpu False --seed 0
                done
              done
            done
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/simple_kt_our_setting_assist2012_fold_0_ob.txt