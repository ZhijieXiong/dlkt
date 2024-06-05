import argparse
import os

import config

from lib.util.data import read_preprocessed_file, write2file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0.txt")
    parser.add_argument("--max_seq_len2delete", type=int, default=50)
    args = parser.parse_args()
    params = vars(args)

    data_dir = os.path.dirname(params["data_path"])
    file_name = os.path.basename(params["data_path"])
    data = read_preprocessed_file(params["data_path"])
    max_seq_len2delete = params["max_seq_len2delete"]

    # 删除正确率极度不平衡的序列
    data_unbiased = []
    num_seq_deleted = 0
    for item_data in data:
        if item_data["seq_len"] >= max_seq_len2delete:
            data_unbiased.append(item_data)
            continue
        correct_seq = item_data["correct_seq"][:item_data["seq_len"]]
        seq_acc = sum(correct_seq) / item_data["seq_len"]
        if 0.1 <= seq_acc <= 0.9:
            data_unbiased.append(item_data)
        else:
            num_seq_deleted += 1

    print(f"num of deleted seq is {num_seq_deleted}")
    write2file(data_unbiased, os.path.join(data_dir, file_name.replace(".txt", f"_unbiased_by_delete_seq.txt")))

