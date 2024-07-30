#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\qdkt_fine_grain_evaluation_our_setting_new_slepemapy.txt"
p1=10
p2=0.4

echo "(${p1}, ${p2}) all"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), PPMCC between history acc and prediction of all" \
  --n 5 --first_num 1

echo "(${p1}, ${p2}) easy"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), PPMCC between history acc and prediction of easy" \
  --n 5 --first_num 1

echo "(${p1}, ${p2}) unbalanced"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p1}, ${p2}), PPMCC between history acc and prediction of unbalanced" \
  --n 5 --first_num 1
