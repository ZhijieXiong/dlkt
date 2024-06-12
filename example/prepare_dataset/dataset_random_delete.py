import argparse
import os
import random

import config

from lib.util.data import read_preprocessed_file, write2file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0.txt")
    parser.add_argument("--max_sample_delete", type=int, default=50)
    args = parser.parse_args()
    params = vars(args)

    data_dir = os.path.dirname(params["data_path"])
    file_name = os.path.basename(params["data_path"])
    data = read_preprocessed_file(params["data_path"])
    max_sample_delete = params["max_sample_delete"]

    # 随机删除序列：不删除均衡序列
    data_deleted = []
    num_deleted = 0
    for item_data in data:
        correct_seq = item_data["correct_seq"][:item_data["seq_len"]]
        seq_acc = sum(correct_seq) / item_data["seq_len"]
        if 0.47 <= seq_acc <= 0.53:
            data_deleted.append(item_data)
        else:
            r = random.random()
            if (r < 0.85) or (num_deleted >= max_sample_delete):
                data_deleted.append(item_data)
            else:
                num_deleted += item_data["seq_len"]

    print(f"num of deleted sample is {num_deleted}")
    write2file(data_deleted, os.path.join(data_dir, file_name.replace(".txt", f"_random_delete_seq.txt")))

