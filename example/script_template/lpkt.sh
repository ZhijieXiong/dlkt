python F:/code/myProjects/dlkt/example/train/lpkt.py \
  --setting_name our_setting --dataset_name ednet-kt1 --data_type only_question --train_file_name ednet-kt1_train_fold_4.txt --valid_file_name ednet-kt1_valid_fold_4.txt --test_file_name ednet-kt1_test_fold_4.txt \
  --optimizer_type adam --weight_decay 0.000001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.002 --enable_lr_schedule True --lr_schedule_type StepLR --lr_schedule_step 15 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --num_concept 188 --num_question 11858 --dim_e 128 --dim_k 128 \
  --dim_correct 50 --dropout 0.2 --save_model False --debug_mode False --seed 0 \
  