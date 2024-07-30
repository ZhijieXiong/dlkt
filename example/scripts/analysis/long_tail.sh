#!/usr/bin/env bash

file_path="F:\code\myProjects\dlkt\example\result_local\our_setting_new\evaluation\auxInfoDct\final\ednet-kt1.txt"

echo "zero shot"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "question zero shot" \
  --n 5 --first_num 0

echo "low frequency"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "question low frequency" \
  --n 5 --first_num 0

echo "middle frequency"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "question middle frequency" \
  --n 5 --first_num 0

echo "high frequency"
python F:/code/myProjects/dlkt/parse_result.py \
  --file_path "${file_path}" \
  --key_words "question high frequency" \
  --n 5 --first_num 0