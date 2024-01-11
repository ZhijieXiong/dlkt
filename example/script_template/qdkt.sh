python F:/code/myProjects/dlkt/example/train/qdkt.py \
  --setting_name random_split_leave_multi_out_setting --dataset_name assist2012 --data_type single_concept --train_file_name assist2012_train_split_5.txt --valid_file_name assist2012_valid_split_5.txt --test_file_name assist2012_test_split_5.txt \
  --optimizer_type adam --weight_decay 0.0001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.001 --enable_lr_schedule True --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --num_concept 265 --num_question 53091 --dim_concept 64 --dim_question 64 \
  --dim_correct 128 --dim_latent 128 --rnn_type gru --num_rnn_layer 1 --dropout 0.1 \
  --num_predict_layer 3 --dim_predict_mid 128 --activate_type relu --use_LLM_emb4question False --use_LLM_emb4concept False \
  --train_LLM_emb True --transfer_head2zero False --head2tail_transfer_method mean_pool --save_model True --seed 0 \
  