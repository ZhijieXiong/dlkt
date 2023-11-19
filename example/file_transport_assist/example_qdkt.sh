#!/bin/bash

python /ghome/xiongzj/code/dlkt/example/train/qdkt.py \
    --setting_name "random_split_leave_multi_out_setting" --data_type "multi_concept" \
    --train_file_name "assist2009_train_split_0.txt" --valid_file_name "assist2009_valid_split_0.txt" --test_file_name "assist2009_test_split_0.txt" \
    --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
    --train_strategy "valid_test" --num_epoch 200 --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
    --main_metric "AUC" --use_multi_metrics False --multi_metrics "[('AUC', 1), ('ACC', 1)]" \
    --learning_rate 0.0003 --train_batch_size 64 --evaluate_batch_size 256 \
    --enable_lr_schedule False --lr_schedule_type "StepLR" --lr_schedule_step 10 --lr_schedule_gamma 0.5 \
    --enable_clip_grad False --grad_clipped 10 \
    --num_concept 123 --num_question 17751 --dim_concept 64 --dim_question 64 --dim_correct 128 --dim_latent 128 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 3 --dim_predict_mid 128 --activate_type "relu" \
    --save_model False --seed 0