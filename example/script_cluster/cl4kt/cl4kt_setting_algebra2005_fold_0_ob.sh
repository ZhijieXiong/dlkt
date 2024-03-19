#!/usr/bin/env bash

{
  dataset_name="algebra2005"
  data_type="single_concept"
  fold=0

  probs1='0.1 0.3'
  probs2='0.1 0.3'
  hard_neg_probs='0.5 1'
  for num_block in 1 2
  do
    for num_head in 4 8
    do
      for prob1 in ${probs1}
      do
        for prob2 in ${probs2}
        do
          for hard_neg_prob in ${hard_neg_probs}
          do
            echo -e "num_block: ${num_block}, num_head: ${num_head}, prob1: ${prob1}, prob2: ${prob2}, hard_neg_prob: ${hard_neg_prob}"
            python /ghome/xiongzj/code/dlkt/example/train/cl4kt.py \
              --setting_name "cl4kt_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
              --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
              --train_strategy "valid_test" --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric "AUC" --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
              --train_batch_size 512 --evaluate_batch_size 1024 \
              --enable_clip_grad False --grad_clipped 10 \
              --num_concept 436 --num_question 173113 \
              --dim_model 64 --key_query_same True --num_head "${num_head}" --num_block "${num_block}" --dim_ff 1024 --dim_final_fc 512 --dropout 0.2 \
              --temp 0.05 --weight_cl_loss 0.1 --use_hard_neg True --hard_neg_weight 1 \
              --aug_order "['mask', 'crop', 'permute', 'replace']" \
              --mask_prob "${prob1}" --crop_prob "${prob1}" --permute_prob "${prob2}" --replace_prob "${prob2}" --hard_neg_prob "${hard_neg_prob}" \
              --save_model False --debug_mode False --use_cpu False --seed 0
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/cl4kt_cl4kt_setting_algebra2005_fold_0_ob.txt