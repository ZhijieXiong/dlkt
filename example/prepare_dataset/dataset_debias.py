import argparse
import os

import config

from lib.util.data import read_preprocessed_file, write2file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0.txt")
    parser.add_argument("--max_num2delete", type=int, default=20)
    args = parser.parse_args()
    params = vars(args)

    data_dir = os.path.dirname(params["data_path"])
    file_name = os.path.basename(params["data_path"])
    data = read_preprocessed_file(params["data_path"])
    max_num_delete = params["max_num2delete"]
    max_seq_len = len(data[0]["mask_seq"])
    for item_data in data:
        if item_data["seq_len"] < (max_num_delete + 3):
            continue
        correct_seq = item_data["correct_seq"][:item_data["seq_len"]]
        seq_acc = sum(correct_seq) / item_data["seq_len"]
        correct_seq_start = correct_seq[:max_num_delete]
        # 删除的策略激进程度由additional_delete控制，该值越大，越激进
        additional_delete = 3
        all_right = sum(correct_seq_start) >= (max_num_delete - additional_delete)
        all_wrong = sum(correct_seq_start) <= additional_delete

        if seq_acc > 0.55:
            if all_right:
                num2delete = max_num_delete
            elif all_wrong:
                num2delete = 0
            else:
                num2delete = correct_seq_start.index(0) + additional_delete
        elif seq_acc < 0.45:
            if all_right:
                num2delete = 0
            elif all_wrong:
                num2delete = max_num_delete
            else:
                num2delete = correct_seq_start.index(1) + additional_delete
        else:
            num2delete = 0
        num2delete = min(num2delete, max_num_delete)
        item_data["seq_len"] -= num2delete

        for k, v in item_data.items():
            if type(v) is list and num2delete > 0:
                item_data[k] = v[num2delete:] + [0] * num2delete

    write2file(data, os.path.join(data_dir, file_name.replace(".txt", f"_unbiased_{max_num_delete}.txt")))

