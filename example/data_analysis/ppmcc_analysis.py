import argparse
import json
import os

import config

from lib.util.data import read_preprocessed_file
from lib.evaluator.util import cal_PPMCC_his_acc_and_cur_label, cal_PPMCC_train_question_acc_and_cur_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2012_train_fold_0.txt")
    parser.add_argument("--train_statics_common_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2012_train_fold_0_statics_common.json")

    args = parser.parse_args()
    params = vars(args)

    data = read_preprocessed_file(params["data_path"])
    for item_data in data:
        if "mask_seq" not in item_data.keys():
            item_data["mask_seq"] = [1] * item_data["seq_len"]
    batch_size = 256
    batch_idx = [i for i in range(0, len(data), batch_size)]
    all_batch = []
    for i_start in batch_idx:
        batch_data = data[i_start: i_start + batch_size]
        seq_lens = list(map(lambda x: x["seq_len"], batch_data))
        max_seq_len = max(seq_lens)
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
        all_batch.append(batch)

    window_lens = [10, 20]
    his_acc_ths = [0.4, 0.3, 0.2]
    print(f"PPMCC between history accuracy and label")
    for his_acc_th in his_acc_ths:
        ppmcc = cal_PPMCC_his_acc_and_cur_label(all_batch, window_lens, his_acc_th)
        for window_len in window_lens:
            for k in ["all", "easy", "normal", "hard", "unbalanced"]:
                print(f"({window_len}, {his_acc_th}), PPMCC between history acc and prediction of {k} is {ppmcc[window_len][k]:.4}")

    # 如果有训练集的习题正确率信息
    if os.path.exists(params["train_statics_common_path"]):
        with open(params["train_statics_common_path"], "r") as f:
            train_statics_common = json.load(f)
        question_acc_dict = {}
        for q_id, q_acc in train_statics_common["question_acc"].items():
            question_acc_dict[int(q_id)] = q_acc
        train_statics_common["question_acc"] = question_acc_dict
        his_acc_ths = [0.4, 0.3, 0.2]
        print(f"\nPPMCC between question accuracy in train dataset and current label")
        for his_acc_th in his_acc_ths:
            ppmcc = cal_PPMCC_train_question_acc_and_cur_label(all_batch, train_statics_common, his_acc_th)
            for k in ["all", "easy", "normal", "hard", "unbalanced"]:
                print(f"({his_acc_th}), PPMCC between question acc and prediction of {k} is {ppmcc[k]:.4}")
