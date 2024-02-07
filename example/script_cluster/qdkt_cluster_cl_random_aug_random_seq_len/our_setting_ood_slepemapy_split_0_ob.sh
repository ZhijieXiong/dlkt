#!/usr/bin/env bash

{
  setting_name="our_setting_ood_by_country"
  dataset_name="slepemapy"
  data_type="single_concept"
  split=0

  weights_decay='0.001 0.0001 0.00001 0'
  temps='0.05'
  weights_cl='0.01 0.1 0.5'
  nums_cluster=(64 256)
  dropouts='0.4'
  for num_cluster in "${nums_cluster[@]}"
  do
    for dropout in ${dropouts}
    do
      for temp in ${temps}
      do
        for weight_decay in ${weights_decay}
        do
          for weight_cl in ${weights_cl}
          do
            echo -e "num_cluster: ${num_cluster}, dropout: ${dropout}, temp: ${temp}, weight_decay: ${weight_decay}, weight_cl: ${weight_cl}"
            python /ghome/xiongzj/code/dlkt/example/train/qdkt_cluster_cl.py \
              --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
              --train_file_name "${dataset_name}_train_split_${split}.txt" --valid_file_name "${dataset_name}_valid_iid_split_${split}.txt" --test_file_name "${dataset_name}_test_ood_split_${split}.txt" \
              --optimizer_type "adam" --weight_decay "${weight_decay}" --momentum 0.9 \
              --train_strategy valid_test --num_epoch 200 \
              --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
              --main_metric "AUC" --use_multi_metrics False \
              --learning_rate 0.001 --enable_lr_schedule False \
              --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
              --enable_clip_grad False --grad_clipped 10.0 \
              --train_batch_size 64 --evaluate_batch_size 256 \
              --num_concept 246 --num_question 5730 \
              --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 1 --dropout "${dropout}" \
              --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
              --num_cluster "${num_cluster}" --temp "${temp}" --weight_cl_loss "${weight_cl}" --latent_type4cl "last_time" \
              --use_warm_up4cl True --epoch_warm_up4cl 3 \
              --use_emb_dropout4cl True --emb_dropout4cl 0.1 \
              --data_aug_type4cl "original_data_aug" --aug_type "random_aug" --use_random_select_aug_len True \
              --mask_prob 0.1 --insert_prob 0.1 --replace_prob 0.2 --crop_prob 0.1 --permute_prob 0.3 \
              --aug_order "['mask', 'crop', 'replace', 'permute']" --offline_sim_type "RCD_graph" \
              --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 3 \
              --use_adv_aug False --epoch_interval_generate 3 --loop_adv 3 --epoch_generate 200 --adv_learning_rate 30.0 --eta 20.0 --gamma 10.0 \
              --save_model False --debug_mode False --seed 0
          done
        done
      done
    done
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_cluster_cl_random_aug_random_seq_len_our_setting_ood_slepemapy_split_0_ob.txt
