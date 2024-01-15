python F:/code/myProjects/dlkt/example/train/akt.py \
  --setting_name our_setting --dataset_name assist2009 --data_type only_question --train_file_name assist2009_train_fold_0.txt --valid_file_name assist2009_valid_fold_0.txt --test_file_name assist2009_test_fold_0.txt \
  --optimizer_type adam --weight_decay 0 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.0001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5, 10] --lr_schedule_gamma 0.5 --train_batch_size 24 --evaluate_batch_size 128 --enable_clip_grad True \
  --grad_clipped 10.0 --num_concept 123 --num_question 17751 --dim_model 256 --key_query_same True \
  --num_head 8 --num_block 2 --dim_ff 256 --dim_final_fc 512 --dropout 0.2 \
  --separate_qa False --seq_representation encoder_output --weight_rasch_loss 0.00001 --save_model False --seed 0 \
  