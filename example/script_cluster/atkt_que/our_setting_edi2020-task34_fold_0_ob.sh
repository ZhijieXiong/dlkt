#!/usr/bin/env bash

{
  dataset_name="edi2020-task34"
  data_type="single_concept"
  fold=0

  dropouts='0.3 0.4 0.5'
  weight_decays='0'
  dims_concept=(64)
  dims_correct=(64)
  dims_latent=(256)
  dims_attention=(256)
  epsilons=(2 5 10)
  betas='0.2 0.5 1'
  for weight_decay in ${weight_decays}
  do
    for dim_attention in "${dims_attention[@]}"
    do
      for dim_concept in "${dims_concept[@]}"
      do
        for dim_correct in "${dims_correct[@]}"
        do
          for dim_latent in "${dims_latent[@]}"
          do
            for epsilon in "${epsilons[@]}"
            do
              for beta in ${betas}
              do
                for dropout in ${dropouts}
                do
                  echo -e "weight_decay: ${weight_decay}, dim_attention: ${dim_attention}, dim_concept: ${dim_concept}, dim_correct: ${dim_correct}, dim_latent: ${dim_latent}, epsilon: ${epsilon}, beta: ${beta}, dropout: ${dropout}"
                  python /ghome/xiongzj/code/dlkt/example/train/atkt.py \
                    --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                    --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                    --optimizer_type adam --weight_decay "${weight_decay}" --momentum 0.9 \
                    --train_strategy valid_test --num_epoch 200 \
                    --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
                    --main_metric AUC --use_multi_metrics False \
                    --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                    --train_batch_size 64 --evaluate_batch_size 256 \
                    --enable_clip_grad False --grad_clipped 10.0 \
                    --use_concept False --num_concept 53 --num_question 948 \
                    --dim_concept "${dim_concept}" --dim_correct "${dim_correct}" --dim_latent "${dim_latent}" \
                    --dim_attention "${dim_attention}" --dropout "${dropout}" --epsilon "${epsilon}" --beta "${beta}" \
                    --save_model False --debug_mode False --seed 0
                done
              done
            done
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/atkt_que_our_setting_edi2020-task34_fold_0_ob.txt

