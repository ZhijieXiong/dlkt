#!/usr/bin/env bash

{
  dataset_name="assist2009"
  data_type="only_question"
  fold=0

  weights_decay='0.0001 0.00001 0'
  weights_cl='0.0001 0.001 0.01 0.1'
  for weight_decay in ${weights_decay}
  do
    for num_cluster in 64 256
    do
      for weight_cl in ${weights_cl}
      do
        echo -e "weight_decay: ${weight_decay}, num_cluster: ${num_cluster}, weight_cl: ${weight_cl}"
        python /ghome/xiongzj/code/dlkt/example/train/qdkt_cluster_cl.py \
          --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
          --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
          --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
          --train_strategy "valid_test" --num_epoch 200 \
          --use_early_stop True --epoch_early_stop 10 \
          --use_last_average False --epoch_last_average 5 \
          --main_metric "AUC" --use_multi_metrics False \
          --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
          --train_batch_size 64 --evaluate_batch_size 256 \
          --enable_clip_grad False --grad_clipped 10.0 \
          --num_concept 123 --num_question 17751 \
          --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
          --use_warm_up4cl True --epoch_warm_up4cl 4 --cl_type "last_time" \
          --num_cluster "${num_cluster}" --temp 0.05 --weight_cl_loss "${weight_cl}" \
          --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
          --aug_type "informative_aug" --use_random_select_aug_len True \
          --mask_prob 0.1 --insert_prob 0.1 --replace_prob 0.1 --crop_prob 0.1 --permute_prob 0.1 \
          --aug_order "['mask', 'insert', 'replace']" --offline_sim_type "RCD_graph" \
          --use_adv_aug False --epoch_interval_generate 3 --loop_adv 3 --epoch_generate 200 --adv_learning_rate 30.0 --eta 20.0 --gamma 10.0 \
          --save_model False --seed 0
      done
    done
  done

} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_cluster_cl_random_aug_random_seq_len_our_setting_assist2009_fold_0_ob.txt
