#!/usr/bin/env bash


dataset_name="assist2017"
data_type="single_concept"
fold=0


#{
#  weight_decays='0.0001 0.00001'
#  dropouts='0.05 0.1 0.15'
#  dims=(48 64 96)
#  for weight_decay in ${weight_decays}
#  do
#    for dim in "${dims[@]}"
#    do
#      for dropout in ${dropouts}
#      do
#        echo -e "weight_decay: ${weight_decay}, dim: ${dim}, dropout: ${dropout}"
#        python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
#          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
#          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
#          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
#          --train_strategy "valid_test" --num_epoch 200 \
#          --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#          --main_metric "AUC" --use_multi_metrics False \
#          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 20 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
#          --train_batch_size 64 --evaluate_batch_size 1024 \
#          --enable_clip_grad False --grad_clipped 10.0 \
#          --num_concept 101 --num_question 2803 --ablation_set 0 \
#          --dim_question "${dim}" --dim_latent "${dim}" --dim_correct 50 --dropout "${dropout}" \
#          --w_que_diff_pred 0 --w_que_disc_pred 0 --w_user_ability_pred 0 --w_penalty_neg 0 --w_learning 0 --w_counter_fact 0 \
#          --save_model False --debug_mode False --use_cpu False --seed 0
#      done
#    done
#  done
#} >> F:/code/myProjects/dlkt/example/result_local/lpkt+_baseline_our_setting_assist2017_fold_0_ob.txt


{
  weight_decays='0.0001'
  dropouts='0.05'
  dims=(64)
  ws_que_diff_pred='0'
  ws_que_disc_pred='0'
  ws_q_able='0.01 0.1 1'
  ws_penalty_neg='0.01 0.1 1'
  ws_learning='0.01 0.1 1'
  ws_counter_fact='0'
  for weight_decay in ${weight_decays}
  do
    for dim in "${dims[@]}"
    do
      for w_que_diff_pred in ${ws_que_diff_pred}
      do
        for w_que_disc_pred in ${ws_que_disc_pred}
        do
          for w_penalty_neg in ${ws_penalty_neg}
          do
            for w_learning in ${ws_learning}
            do
              for w_q_able in ${ws_q_able}
              do
                for w_counter_fact in ${ws_counter_fact}
                do
                  for dropout in ${dropouts}
                  do
                    echo -e "weight_decay: ${weight_decay}, dim: ${dim}, dropout: ${dropout}"
                    echo -e "w_penalty_neg: ${w_penalty_neg}, w_learning: ${w_learning}, w_counter_fact: ${w_counter_fact}, , w_q_able: ${w_q_able}, w_que_diff_pred: ${w_que_diff_pred}, w_que_disc_pred: ${w_que_disc_pred} "
                    python F:/code/myProjects/dlkt/example/train/lpkt_plus.py \
                      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
                      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
                      --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
                      --train_strategy "valid_test" --num_epoch 200 \
                      --use_early_stop True --epoch_early_stop 20 --use_last_average False --epoch_last_average 5 \
                      --main_metric "AUC" --use_multi_metrics False \
                      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 20 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
                      --train_batch_size 64 --evaluate_batch_size 1024 \
                      --enable_clip_grad False --grad_clipped 10.0 \
                      --num_concept 101 --num_question 2803 --ablation_set 0 \
                      --multi_stage True --user_weight_init False --que_weight_init True \
                      --dim_question "${dim}" --dim_latent "${dim}" --dim_correct 50 --dropout "${dropout}" \
                      --min_fre4diff 50 --min_fre4disc 50 --min_seq_len4disc 20 --percent_threshold 0.27 \
                      --w_que_diff_pred "${w_que_diff_pred}" --w_que_disc_pred "${w_que_disc_pred}" --w_user_ability_pred 0 \
                      --w_penalty_neg "${w_penalty_neg}" --w_learning "${w_learning}" --w_counter_fact "${w_counter_fact}" --w_q_table "${w_q_able}" \
                      --save_model False --debug_mode False --use_cpu False --seed 0
                  done
                done
              done
            done
          done
        done
      done
    done
  done
} >> F:/code/myProjects/dlkt/example/result_local/lpkt+-penalty_neg-learn-q_table-our_setting_assist2017_fold_0_ob1.txt