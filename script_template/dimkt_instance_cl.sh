python F:/code/myProjects/dlkt/example/train/dimkt_instance_cl.py \
  --setting_name random_split_leave_multi_out_setting --dataset_name assist2012 --data_type single_concept --train_file_name assist2012_train_split_5.txt --valid_file_name assist2012_valid_split_5.txt --test_file_name assist2012_test_split_5.txt \
  --optimizer_type adam --weight_decay 0.0001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --num_min_question 25 --num_min_concept 30 --num_concept 265 --num_question 53091 \
  --dim_emb 128 --num_question_diff 100 --num_concept_diff 100 --dropout 0.2 --temp 0.05 \
  --weight_cl_loss 0.1 --use_warm_up4cl False --epoch_warm_up4cl 2 --use_stop_cl_after False --epoch_stop_cl 3 \
  --use_weight_dynamic False --weight_dynamic_type multi_step --multi_step_weight "[[1, 0.1], [3, 0.03], [5, 0.01], [10, 0.0001], [200, 0.000001]]" --linear_increase_epoch 1 --linear_increase_value 0.1 \
  --latent_type4cl last_time --use_emb_dropout4cl True --emb_dropout4cl 0.1 --data_aug_type4cl original_data_aug --use_neg True \
  --use_neg_filter True --neg_sim_threshold 0.8 --offline_sim_type order --use_online_sim True --use_warm_up4online_sim True \
  --epoch_warm_up4online_sim 4 --aug_type informative_aug --use_random_select_aug_len False --aug_order "['crop', 'replace', 'insert']" --mask_prob 0.1 \
  --insert_prob 0.1 --replace_prob 0.1 --crop_prob 0.1 --permute_prob 0.1 --use_hard_neg False \
  --hard_neg_prob 1 --use_adv_aug False --epoch_interval_generate 1 --loop_adv 3 --epoch_generate 40 \
  --adv_learning_rate 20.0 --eta 5.0 --gamma 1.0 --save_model True --seed 0 \
  