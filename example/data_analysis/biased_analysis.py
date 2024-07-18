import argparse
import json
import os

import config

from lib.util.data import read_preprocessed_file
from lib.evaluator.util import get_num_seq_fine_grained_sample, get_num_question_fine_grained_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_ood_by_country\slepemapy_test_ood_split_0.txt")
    parser.add_argument("--train_statics_common_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_ood_by_country\slepemapy_train_split_0_statics_common.json")
    args = parser.parse_args()
    params = vars(args)

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
            "label_seqs": [],
            "mask_seqs": [],
            "predict_score_seqs": []
        }
        for item_data in batch_data:
            batch["question_seqs"].append(item_data["question_seq"][:-1])
            batch["label_seqs"].append(item_data["correct_seq"][:-1])
            batch["mask_seqs"].append(item_data["mask_seq"][:-1])
            batch["predict_score_seqs"].append([1] * max_seq_len)
            num_sample_all += item_data["seq_len"]
        all_batch.append(batch)

    # 数据集划分：不同参数组合
    sub_data_statics = {}
    for window_len in [10, 15, 20]:
        for acc_th in [0.4, 0.3, 0.2]:
            k = f"({window_len}, {acc_th})"
            num_easy, num_normal, num_hard, num_hard_label0, num_hard_label1 = \
                get_num_seq_fine_grained_sample(all_batch, window_len, acc_th)
            sub_data_statics[k] = {
                "seq easy sample": num_easy / num_sample_all,
                "seq hard sample": num_hard / num_sample_all,
                "seq normal sample": num_normal / num_sample_all,
                "seq hard sample with label 1": num_hard_label1 / num_hard
            }

            for kk in sub_data_statics[k]:
                print(f"{k}, proportion of {kk} is {sub_data_statics[k][kk]:<6.5f}")

    # 如果有训练集的习题正确率信息
    if os.path.exists(params["train_statics_common_path"]):
        with open(params["train_statics_common_path"], "r") as f:
            train_statics_common = json.load(f)
        question_acc_dict = {}
        for q_id, q_acc in train_statics_common["question_acc"].items():
            question_acc_dict[int(q_id)] = q_acc
        train_statics_common["question_acc"] = question_acc_dict

        for acc_th in [0.4, 0.3, 0.2]:
            k = f"({acc_th})"
            num_easy, num_normal, num_hard, num_hard_label0, num_hard_label1 = \
                get_num_question_fine_grained_sample(all_batch, train_statics_common, acc_th)
            sub_data_statics[k] = {
                "question easy sample": num_easy / num_sample_all,
                "question hard sample": num_hard / num_sample_all,
                "question normal sample": num_normal / num_sample_all,
                "question hard label with label 1": num_hard_label1 / num_hard
            }

            for kk in sub_data_statics[k]:
                print(f"{k}, proportion of {kk} is {sub_data_statics[k][kk]:<6.5f}")
