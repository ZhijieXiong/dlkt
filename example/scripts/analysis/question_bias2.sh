#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\akt-IPS-double-3e-1_ood_setting_slepemapy_save.txt"
test_or_valid="valid"
p=0.4

echo "(${p}, ) easy"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}, ) ${test_or_valid} question easy point" \
  --n 5 --first_num 1

echo "(${p}, ) normal"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}, ) ${test_or_valid} question normal point" \
  --n 5 --first_num 1

echo "(${p}, ) hard"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(${p}, ) ${test_or_valid} question hard point" \
  --n 5 --first_num 1