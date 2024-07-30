#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\akt-IPS-double-3e-1_ood_setting_slepemapy_save.txt"
test_or_valid="valid"

echo "(10, 0.4) easy"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(10, 0.4) ${test_or_valid} seq easy point" \
  --n 5 --first_num 1

echo "(10, 0.4) normal"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(10, 0.4) ${test_or_valid} seq normal point" \
  --n 5 --first_num 1

echo "(10, 0.4) hard"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "(10, 0.4) ${test_or_valid} seq hard point" \
  --n 5 --first_num 1

#echo "(20, 0.4) easy"
#python F:/code/myProjects/dlkt/parse_result.py \
#  --file_path "${file_path}" \
#  --key_words "(20, 0.4) ${test_or_valid} seq easy point" \
#  --n 5 --first_num 1

#echo "(20, 0.4) normal"
#python F:/code/myProjects/dlkt/parse_result.py \
#  --file_path "${file_path}" \
#  --key_words "(20, 0.4) ${test_or_valid} seq normal point" \
#  --n 5 --first_num 1

#echo "(20, 0.4) hard"
#python F:/code/myProjects/dlkt/parse_result.py \
#  --file_path "${file_path}" \
#  --key_words "(20, 0.4) ${test_or_valid} seq hard point" \
#  --n 5 --first_num 1
