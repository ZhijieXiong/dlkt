#!/usr/bin/env bash

{
  dataset_name="xes3g5m"
  data_type="only_question"

  for fold in 0 1 2 3 4
  do
    echo -e "fold: ${fold}"
    python /ghome/xiongzj/code/dlkt-release/example/train/akt_que.py \
      --setting_name "our_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train_fold_${fold}.txt" --valid_file_name "${dataset_name}_valid_fold_${fold}.txt" --test_file_name "${dataset_name}_test_fold_${fold}.txt" \
      --optimizer_type "adam" --weight_decay 0 --momentum 0.9 \
      --train_strategy "valid_test" --num_epoch 200 \
      --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 \
      --main_metric "AUC" --use_multi_metrics False \
      --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type "MultiStepLR" --lr_schedule_step 10 --lr_schedule_milestones "[5, 10]" --lr_schedule_gamma 0.5 \
      --train_batch_size 24 --evaluate_batch_size 128 \
      --enable_clip_grad True --grad_clipped 10.0 \
      --num_question 7652 --que_emb_file_name "qid2content_emb.json" --frozen_que_emb False \
      --dim_model 256 --key_query_same True --num_head 8 --num_block 4 --dim_ff 256 --dim_final_fc 512 --dropout 0.2 \
      --separate_qa False --seq_representation "encoder_output" --weight_rasch_loss 0.00001 \
      --use_sample_reweight False --save_model True --debug_mode False --use_cpu False --seed 0 --trace_epoch True
  done

} >> /ghome/xiongzj/code/dlkt-release/example/result_cluster/our_setting_akt_que_official_emb_free_emb_xes3g5m_save.txt