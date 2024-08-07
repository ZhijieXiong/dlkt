import argparse
import pandas as pd
import os

import config

from lib.util.data import read_preprocessed_file
from lib.evaluator.util import get_seq_fine_grained_sample_mask, get_question_fine_grained_sample_mask
from lib.data_processor.load_raw import load_csv


# 从测试集中找history bias-conflicting的case，并获取concept文本信息
# 目前仅支持single_concept
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"/Users/dream/myProjects/dlkt/lab/settings/our_setting_new/assist2012_test_fold_0.txt")
    parser.add_argument("--train_statics_common_path", type=str,
                        default=r"/Users/dream/myProjects/dlkt/lab/settings/our_setting_new/assist2012_train_fold_0_statics_common.json")
    parser.add_argument("--concept_id_map_path", type=str,
                        default=r"/Users/dream/myProjects/dlkt/lab/dataset_preprocessed/assist2012/concept_id_map_single_concept.csv")
    parser.add_argument("--concept_id2name_map_path", type=str,
                        default=r"/Users/dream/myProjects/dlkt/lab/dataset_preprocessed/assist2012/concept_id2name_map.csv")
    args = parser.parse_args()
    params = vars(args)

    concept_id_map = load_csv(params["concept_id_map_path"])
    concept_id2name_map = load_csv(params["concept_id2name_map_path"])
    # 合并DataFrame
    merged_df = pd.merge(concept_id_map, concept_id2name_map, on='concept_id', how='left')
    concept_id2name = merged_df.dropna().set_index('concept_mapped_id')['concept_name'].to_dict()

    # 提取需要的列并转换为字典
    result = merged_df.dropna().set_index('concept_mapped_id')['concept_name'].to_dict()

    data = read_preprocessed_file(params["data_path"])
    batch_size = 256
    batch_idx = [i for i in range(0, len(data), batch_size)]
    all_batch = []
    max_seq_len = len(data[0]["mask_seq"]) - 1
    num_sample_all = 0
    for i_start in batch_idx:
        batch_data = data[i_start: i_start + batch_size]
        batch = {
            "question_seqs": [],
            "concept_seqs": [],
            "label_seqs": [],
            "mask_seqs": [],
            "predict_score_seqs": []
        }
        for item_data in batch_data:
            batch["question_seqs"].append(item_data["question_seq"])
            batch["concept_seqs"].append(item_data["concept_seq"])
            batch["label_seqs"].append(item_data["correct_seq"])
            batch["mask_seqs"].append(item_data["mask_seq"])
            batch["predict_score_seqs"].append([1] * max_seq_len)
            num_sample_all += item_data["seq_len"]
        all_batch.append(batch)

    window_len = 10
    acc_th = 0.2
    bias_conflicting_examples = []
    for batch in all_batch:
        seq_easy_mask, seq_normal_mask, seq_hard_mask = get_seq_fine_grained_sample_mask(batch, window_len, acc_th)
        for i, mask_seq in enumerate(seq_hard_mask):
            question_seq = batch["question_seqs"][i]
            correct_seq = batch["label_seqs"][i]
            for j, mask in enumerate(mask_seq):
                if mask:
                    bias_conflicting_example = {
                        "history_question_seq": question_seq[:j],
                        "history_correct_seq": correct_seq[:j],
                        "current_question": question_seq[j],
                        "current_correct": correct_seq[j]
                    }
                    bias_conflicting_examples.append(bias_conflicting_example)

