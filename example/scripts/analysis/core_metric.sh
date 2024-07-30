#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\qdkt-IPS-double_fine_grain_evaluation_our_setting_new_xes3g5m.txt"

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "CORE metric allow repeat" \
  --n 5 --first_num 0
