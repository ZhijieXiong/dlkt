#!/usr/bin/env bash


python F:/code/myProjects/dlkt/example/generate_ednet_raw.py \
  --dataset_src_dir "E:\dataSet\knowledgeTracingtData\EDnet\EdNet-KT1\KT1" \
  --contents_dir "E:\dataSet\knowledgeTracingtData\EDnet\EdNet-Contents\contents"

python F:/code/myProjects/dlkt/example/preprocess.py --dataset_name "ednet-kt1"