#!/usr/bin/env bash

setting_name="pykt_question_setting"

{
  dataset_name="assist2009"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0.0001 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 123 --num_question 17751 \
      --dim_emb 256 --dim_latent 256 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 2 --dim_predict_mid 256 --activate_type "sigmoid" \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_assist2009_save.txt


{
  dataset_name="bridge2algebra2006"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type adam --weight_decay 0.0001 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 493 --num_question 129263 \
      --dim_emb 64 --dim_latent 256 --rnn_type gru --num_rnn_layer 1 --dropout 0.5 --num_predict_layer 2 --dim_predict_mid 512 --activate_type sigmoid \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_bridge2algebra2006_save.txt


{
  dataset_name="algebra2005"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type adam --weight_decay 0 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 112 --num_question 173113 \
      --dim_emb 256 --dim_latent 256 --rnn_type gru --num_rnn_layer 1 --dropout 0.3 --num_predict_layer 2 --dim_predict_mid 256 --activate_type sigmoid \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_algebra2005_save.txt


{
  dataset_name="edi2020-task34"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type adam --weight_decay 0 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 53 --num_question 948 \
      --dim_emb 64 --dim_latent 256 --rnn_type gru --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 2 --dim_predict_mid 512 --activate_type sigmoid \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_edi2020-task34_save.txt


{
  dataset_name="statics2011"
  data_type="single_concept"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 27 --num_question 1223 \
      --dim_emb 64 --dim_latent 256 --rnn_type "gru" --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 1 --dim_predict_mid 256 --activate_type "sigmoid" \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_statics2011_save.txt


{
  dataset_name="xes3g5m"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python F:/code/myProjects/dlkt/example/train/dkt.py \
      --setting_name "${setting_name}" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test.txt" \
      --optimizer_type adam --weight_decay 0 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric AUC --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 128 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept True --num_concept 865 --num_question 7652 \
      --dim_emb 64 --dim_latent 64 --rnn_type gru --num_rnn_layer 1 --dropout 0.1 --num_predict_layer 2 --dim_predict_mid 512 --activate_type sigmoid \
      --save_model True --debug_mode False --use_cpu False --seed 0
  done
} >> F:/code/myProjects/dlkt/example/result_local/dkt_pykt_question_setting_xes3g5m_save.txt