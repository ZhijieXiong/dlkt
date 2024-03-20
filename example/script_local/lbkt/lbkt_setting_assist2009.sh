#!/usr/bin/env bash


{
  dataset_name="assist2009"
  data_type="only_question"

  for seed in 0 1 2 3 4
  do
    echo -e "seed: ${seed}"
    python F:/code/myProjects/dlkt/example/train/lbkt.py \
      --setting_name "lbkt_setting" --dataset_name "${dataset_name}" --data_type "${data_type}" \
      --train_file_name "${dataset_name}_train.txt" --valid_file_name "${dataset_name}_valid.txt" --test_file_name "${dataset_name}_test.txt" \
      --num_concept 123 --num_question 17751 \
      --save_model True --debug_mode False --use_cpu False --seed "${seed}"
  done
} >> F:/code/myProjects/dlkt/example/result_local/lbkt_lbkt_setting_assist2009_save.txt
