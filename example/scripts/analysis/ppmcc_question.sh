#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\qdkt_fine_grain_evaluation_our_setting_new_slepemapy.txt"
p=0.4

echo "(${p}) all"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), PPMCC between question acc and prediction of all" \
  --n 5 --first_num 1

echo "(${p}) easy"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), PPMCC between question acc and prediction of easy" \
  --n 5 --first_num 1

echo "(${p}) unbalanced"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}), PPMCC between question acc and prediction of unbalanced" \
  --n 5 --first_num 1
