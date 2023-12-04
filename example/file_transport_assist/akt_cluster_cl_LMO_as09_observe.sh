#!/usr/bin/env bash

{
    dataset_name="assist2009"
    domain=6

#    ob1
#    echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 128, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 128 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0
#
#    echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 256, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0
#
#    echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 512, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 512 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0
#
#    echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 1024, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 1024 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0



#    # ob2
#    echo -e "seed: 0, temp 0.01, weight 0.001, num cluster 256, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 256 --temp 0.01 --weight_cl_loss 0.001 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0
#
#    echo -e "seed: 0, temp 0.01, weight 0.005, num cluster 256, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 256 --temp 0.01 --weight_cl_loss 0.005 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0
#
#
#    echo -e "seed: 0, temp 0.01, weight 0.05, num cluster 256, last time"
#    python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#      --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#      --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#      --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#      --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#      --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#      --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#      --enable_clip_grad True --grad_clipped 10 \
#      --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#      --seq_representation "encoder_output" \
#      --num_cluster 256 --temp 0.01 --weight_cl_loss 0.05 --cl_type "last_time" \
#      --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#      --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#      --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#      --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#      --save_model False --seed 0


  # ob3
#  echo -e "seed: 0, temp 0.01, weight 0.001, num cluster 256, mean_pool"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.001 --cl_type "mean_pool" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 0
#
#  echo -e "seed: 0, temp 0.01, weight 0.005, num cluster 256, mean_pool"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.005 --cl_type "mean_pool" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 0
#
#  echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 256, mean_pool"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "mean_pool" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 0
#
#
#  echo -e "seed: 0, temp 0.01, weight 0.05, num cluster 256, mean_pool"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.05 --cl_type "mean_pool" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 0

  # ob4
#  echo -e "seed: 1, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.1"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.1 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 1
#
#  echo -e "seed: 1, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.2"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.2 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 1
#
#  echo -e "seed: 1, temp 0.01, weight 0.01, num cluster 256, last_time"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 1
#
#  echo -e "seed: 2, temp 0.01, weight 0.01, num cluster 256, last_time"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 2
#
#  echo -e "seed: 3, temp 0.01, weight 0.01, num cluster 256, last_time"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.3 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 3

#  echo -e "seed: 2, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.2"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.2 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 2
#
#  echo -e "seed: 3, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.2"
#  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
#    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
#    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
#    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
#    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
#    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
#    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
#    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
#    --enable_clip_grad True --grad_clipped 10 \
#    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
#    --seq_representation "encoder_output" \
#    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
#    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
#    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
#    --mask_prob 0.2 --insert_prob 0.2 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
#    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
#    --save_model False --seed 3


  echo -e "seed: 0, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.1"
  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
    --enable_clip_grad True --grad_clipped 10 \
    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
    --seq_representation "encoder_output" \
    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
    --mask_prob 0.2 --insert_prob 0.1 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
    --save_model False --seed 0

  echo -e "seed: 1, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.1"
  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
    --enable_clip_grad True --grad_clipped 10 \
    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
    --seq_representation "encoder_output" \
    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
    --mask_prob 0.2 --insert_prob 0.1 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
    --save_model False --seed 1

  echo -e "seed: 2, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.1"
  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
    --enable_clip_grad True --grad_clipped 10 \
    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
    --seq_representation "encoder_output" \
    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
    --mask_prob 0.2 --insert_prob 0.1 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
    --save_model False --seed 2

  echo -e "seed: 3, temp 0.01, weight 0.01, num cluster 256, last_time, insert 0.1"
  python F:/code/myProjects/dlkt/example/train/akt_cluster_cl.py \
    --setting_name "random_split_leave_multi_out_setting" --dataset_name "${dataset_name}" --data_type "multi_concept" \
    --train_file_name "${dataset_name}_train_split_${domain}.txt" --valid_file_name "${dataset_name}_valid_split_${domain}.txt" --test_file_name "${dataset_name}_test_split_${domain}.txt" \
    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
    --learning_rate 0.0004 --train_batch_size 32 --evaluate_batch_size 256 \
    --enable_lr_schedule True --lr_schedule_type "MultiStepLR" --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
    --enable_clip_grad True --grad_clipped 10 \
    --num_concept 123 --num_question 17751 --dim_model 64 --key_query_same True --num_head 8 --num_block 2 --dim_ff 128 --dim_final_fc 256 --dropout 0.1 --separate_qa False --weight_rasch_loss 0.00001 \
    --seq_representation "encoder_output" \
    --num_cluster 256 --temp 0.01 --weight_cl_loss 0.01 --cl_type "last_time" \
    --use_warm_up4cluster_cl True --epoch_warm_up4cluster_cl 4 --use_online_sim True --use_warm_up4online_sim True --epoch_warm_up4online_sim 4 \
    --aug_type "informative_aug" --use_random_select_aug_len True --aug_order "['crop', 'replace', 'insert']" --offline_sim_type "order" \
    --mask_prob 0.2 --insert_prob 0.1 --replace_prob 0.3 --crop_prob 0.1 --permute_prob 0.2 --hard_neg_prob 1 \
    --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 --adv_learning_rate 10 --eta 5 --gamma 1 \
    --save_model False --seed 3

} >> F:/code/myProjects/dlkt/example/result_local/akt_cluster_cl_LMO_as09_ob5.txt