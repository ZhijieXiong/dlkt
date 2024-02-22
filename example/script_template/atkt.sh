python F:/code/myProjects/dlkt/example/train/atkt.py \
  --setting_name our_setting --dataset_name edi2020-task34 --data_type single_concept --train_file_name edi2020-task34_train_fold_0.txt --valid_file_name edi2020-task34_valid_fold_0.txt --test_file_name edi2020-task34_test_fold_0.txt \
  --optimizer_type adam --weight_decay 0.000001 --momentum 0.9 --train_strategy valid_test --num_epoch 200 \
  --use_early_stop True --epoch_early_stop 10 --use_last_average False --epoch_last_average 5 --main_metric AUC \
  --use_multi_metrics False --learning_rate 0.001 --enable_lr_schedule False --lr_schedule_type MultiStepLR --lr_schedule_step 10 \
  --lr_schedule_milestones [5] --lr_schedule_gamma 0.5 --train_batch_size 64 --evaluate_batch_size 256 --enable_clip_grad False \
  --grad_clipped 10.0 --use_concept False --num_concept 53 --num_question 948 --dim_concept 64 \
  --dim_correct 64 --dim_latent 256 --dim_attention 64 --dropout 0.1 --epsilon 15 \
  --beta 2 --save_model False --debug_mode False --seed 0 