#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\simpleKT_fine_grain_evaluation_our_setting_new_xes3g5m.txt"
p1=10
p2=0.4

echo "seq easy samples"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}) seq easy samples" \
  --n 5 --first_num 1

echo "seq normal samples"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}) seq normal samples" \
  --n 5 --first_num 1

echo "seq hard samples"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}) seq hard samples" \
  --n 5 --first_num 1