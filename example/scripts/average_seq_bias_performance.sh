#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\auxInfoDct-baseline_our_setting_new_assist2009_save.txt"
test_or_valid="test"

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of non seq easy samples ((20, 0.4))" \
  --n 5 --first_num 1

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of non seq easy samples ((30, 0.3))" \
  --n 5 --first_num 1

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of non seq easy samples ((40, 0.2))" \
  --n 5 --first_num 1

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of seq biased samples ((20, 0.4))" \
  --n 5 --first_num 1

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of seq biased samples ((30, 0.3))" \
  --n 5 --first_num 1

python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "${test_or_valid} performance of seq biased samples ((40, 0.2))" \
  --n 5 --first_num 1