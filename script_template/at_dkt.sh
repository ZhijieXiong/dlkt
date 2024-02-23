python /Users/dream/Desktop/code/projects/dlkt/example/train/at_dkt.py \
  --setting_name our_setting --dataset_name assist2012 --data_type single_concept --train_file_name assist2012_train_fold_0.txt --valid_file_name assist2012_valid_fold_0.txt --test_file_name assist2012_test_fold_0.txt \
  --optimizer_type adam --weight_decay 0.0001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --num_concept 265 --num_question 53091 --dim_emb 256 --dim_latent 256 \
  --rnn_type lstm --num_rnn_layer 1 --dropout 0.3 --QT_net_type transformer --QT_rnn_type lstm \
  --QT_num_rnn_layer 4 --QT_transformer_num_block 1 --QT_transformer_num_head 4 --IK_start 120 --weight_QT_loss 1 \
  --weight_IK_loss 0.5 --save_model False --debug_mode False --seed 0 