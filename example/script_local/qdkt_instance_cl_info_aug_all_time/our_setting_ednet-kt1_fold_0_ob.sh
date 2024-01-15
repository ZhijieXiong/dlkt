#!/usr/bin/env bash

{
  dataset_name="ednet-kt1"
  data_type="only_question"
  fold=0

  weights_cl='0.0001 0.001 0.01 0.1'
  for weight_cl in ${weights_cl}
  do
    echo -e "weight_cl: ${weight_cl}"
    python /ghome/xiongzj/code/dlkt/example/train/qdkt_instance_cl.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --num_concept 188 --num_question 11858 \
      --dim_concept 64 --dim_question 64 --dim_correct 64 --dim_latent 64 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.2 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
      --cl_space "latent" --temp 0.05 --weight_cl_loss "${weight_cl}" --latent_type4cl "all_time" \
      --num2drop_question4dis 30 --num2drop_concept4dis 500 --min_seq_len4dis 20 --dis_threshold 0.3 \
      --use_warm_up4cl False --epoch_warm_up4cl 2 --use_stop_cl_after False --epoch_stop_cl 3 \
      --use_weight_dynamic False --weight_dynamic_type "multi_step" --multi_step_weight "[[1, 0.1], [3, 0.03], [5, 0.01], [10, 0.0001], [200, 0.000001]]" --linear_increase_epoch 1 --linear_increase_value 0.1 \
      --use_emb_dropout4cl True --emb_dropout4cl 0.1 \
      --data_aug_type4cl "original_data_aug" --use_neg True --use_neg_filter False --neg_sim_threshold 0.8 \
      --aug_type "informative_aug" --use_random_select_aug_len True \
      --mask_prob 0.1 --insert_prob 0.1 --replace_prob 0.1 --crop_prob 0.1 --permute_prob 0.1 --aug_order "['mask', 'replace', 'insert']" \
      --offline_sim_type "RCD_graph" --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
      --use_hard_neg False --hard_neg_prob 1 \
      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 20.0 --eta 5.0 --gamma 1.0 \
      --save_model False --seed 0
  done
} >> /ghome/xiongzj/code/dlkt/example/result_cluster/qdkt_instance_cl_info_aug_all_time_our_setting_ednet-kt1_fold_0_ob.txt
