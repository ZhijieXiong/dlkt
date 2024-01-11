python F:/code/myProjects/dlkt/example/train/dkt.py \
  --setting_name our_setting --dataset_name assist2009 --data_type only_question --train_file_name assist2009_train_fold_0.txt --valid_file_name assist2009_valid_fold_0.txt --test_file_name assist2009_test_fold_0.txt \
  --optimizer_type adam --weight_decay 0.0001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 128 --enable_clip_grad False \
  --grad_clipped 10.0 --num_concept 123 --num_question 17751 --dim_emb 256 --dim_latent 256 \
  --rnn_type gru --num_rnn_layer 1 --dropout 0.2 --num_predict_layer 2 --dim_predict_mid 256 \
  --activate_type sigmoid --use_concept True --save_model False --seed 0 