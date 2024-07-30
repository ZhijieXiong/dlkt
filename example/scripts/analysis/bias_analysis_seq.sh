#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\train_dataset_bias_analysis_statics2011.txt"
p1=10
p2=0.4

echo "(${p1}, ${p2}) proportion of seq easy sample"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), proportion of seq easy sample is" \
  --n 5 --first_num 1

echo "(${p1}, ${p2}) proportion of seq hard sample"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), proportion of seq hard sample is" \
  --n 5 --first_num 1

echo "(${p1}, ${p2}) proportion of seq hard sample with label 1 is"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), proportion of seq hard sample with label 1" \
  --n 5 --first_num 1
