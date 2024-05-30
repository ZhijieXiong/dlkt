import argparse

import config

from lib.util.data import read_preprocessed_file
from lib.evaluator.util import get_seq_easy_point, get_seq_biased_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\settings\our_setting_new\assist2009_train_fold_0.txt")
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
            num_sample_all += item_data["seq_len"] - 1
        all_batch.append(batch)

    # 数据集划分：不同参数组合
    sub_data_statics = {}
    for previous_seq_len in [20, 30, 40]:
        for seq_most_accuracy in [0.4, 0.3, 0.2]:
            k = f"({previous_seq_len}, {seq_most_accuracy})"
            seq_easy, non_seq_easy = get_seq_easy_point(all_batch, previous_seq_len, seq_most_accuracy)
            seq_hard = get_seq_biased_point(all_batch, previous_seq_len, seq_most_accuracy)
            num_easy = len(seq_easy["high_acc_and_right"]["question"]) + len(seq_easy["low_acc_and_wrong"]["question"])
            num_non_easy = len(non_seq_easy["predict_score"])
            num_hard = len(seq_hard["high_acc_but_wrong"]["question"]) + len(seq_hard["low_acc_but_right"]["question"])
            sub_data_statics[k] = {
                "seq_easy_sample": num_easy / num_sample_all,
                "non_seq_easy_sample": num_non_easy / num_sample_all,
                "seq_hard_sample": num_hard / num_sample_all
            }

            print(f"bias params is ({previous_seq_len}, {seq_most_accuracy}), "
                  f"proportion of seq easy sample is {sub_data_statics[k]['seq_easy_sample']:<6.5f}, "
                  f"proportion of non seq easy sample is {sub_data_statics[k]['non_seq_easy_sample']:<6.5f}, "
                  f"proportion of seq hard sample is {sub_data_statics[k]['seq_hard_sample']:<6.5f}")
