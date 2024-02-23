python F:/code/myProjects/dlkt/example4cognitive_diagnosis/train/ncd.py \
  --setting_name ncd_setting --dataset_name assist2009 --train_file_name assist2009_train_fold_0.txt --valid_file_name assist2009_valid_fold_0.txt --test_file_name assist2009_test_fold_0.txt --optimizer_type adam \
  --weight_decay 0 --momentum 0.9 --train_strategy valid_test --num_epoch 50 --use_early_stop True \
  --epoch_early_stop 5 --use_last_average False --epoch_last_average 5 --main_metric AUC --use_multi_metrics False \
  --learning_rate 0.002 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 --lr_schedule_milestones [5] \
  --lr_schedule_gamma 0.5 --train_batch_size 32 --evaluate_batch_size 256 --enable_clip_grad False --grad_clipped 10.0 \
  --num_user 2500 --num_concept 123 --num_question 17751 --dim_predict1 512 --dim_predict2 256 \
  --dropout 0.5 --save_model False --use_cpu False --debug_mode False --seed 0 \
  