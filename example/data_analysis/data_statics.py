import argparse
import os

import config

from lib.util.data import read_preprocessed_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new")
    parser.add_argument("--data_names", type=str,
                        default="['ednet-kt1_train_fold_0_unbiased_by_delete_seq.txt', 'ednet-kt1_train_fold_1_unbiased_by_delete_seq.txt', 'ednet-kt1_train_fold_2_unbiased_by_delete_seq.txt', 'ednet-kt1_train_fold_3_unbiased_by_delete_seq.txt', 'ednet-kt1_train_fold_4_unbiased_by_delete_seq.txt']")
    args = parser.parse_args()
    params = vars(args)

    num_sample_all = 0
    data_paths = list(map(lambda x: os.path.join(params["data_dir"], x), eval(params["data_names"])))
    for data_path in data_paths:
        data = read_preprocessed_file(data_path)
        for item_data in data:
            num_sample_all += item_data["seq_len"]
    print(f"ave num of sample: {num_sample_all / len(data_paths)}")

