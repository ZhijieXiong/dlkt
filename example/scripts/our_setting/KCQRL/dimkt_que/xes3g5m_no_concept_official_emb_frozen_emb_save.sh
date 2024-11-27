#!/usr/bin/env bash

{
  dataset_name="xes3g5m"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python /ghome/xiongzj/code/dlkt-release/example/train/dimkt_que.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0.00001 --momentum 0.9 \
      --train_strategy valid_test --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 \
      --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.001 --enable_lr_schedule True --lr_schedule_type "StepLR" --lr_schedule_step 5 --lr_schedule_milestones "[5]" --lr_schedule_gamma 0.5 \
      --train_batch_size 64 --evaluate_batch_size 256 \
      --enable_clip_grad False --grad_clipped 10.0 \
      --use_concept False --que_emb_file_name "qid2content_emb.json" --frozen_que_emb True \
      --num_min_question 30 --num_min_concept 30 --num_concept 865 --num_question 7652 \
      --dim_emb 128 --num_question_diff 100 --num_concept_diff 100 --dropout 0.2 \
      --use_sample_reweight False --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done
} >> /ghome/xiongzj/code/dlkt-release/example/result_cluster/our_setting_dimkt_que_no_concept_official_emb_frozen_emb_xes3g5m_save.txt