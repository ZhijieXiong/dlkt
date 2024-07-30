#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\train_dataset_bias_analysis_statics2011.txt"
p=0.4

echo "(${p}) proportion of question easy sample"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), proportion of question easy sample is" \
  --n 5 --first_num 1

echo "(${p}) proportion of question hard sample"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), proportion of question hard sample is" \
  --n 5 --first_num 1

echo "(${p}) proportion of question hard sample with label 1"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), proportion of question hard label with label 1 is" \
  --n 5 --first_num 1
