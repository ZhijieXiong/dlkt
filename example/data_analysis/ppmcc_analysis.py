import argparse

import config

from lib.util.data import read_preprocessed_file
from lib.evaluator.util import cal_PPMCC_label


if __name__ == "__main__":
    # 只能用来分析整体数据集（即无mask的数据集）
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\ednet-kt1\data_only_question.txt")
    args = parser.parse_args()
    params = vars(args)

    data = read_preprocessed_file(params["data_path"])
    for item_data in data:
        if "mask_seq" not in item_data.keys():
            item_data["mask_seq"] = [1] * item_data["seq_len"]
    batch_size = 256
    batch_idx = [i for i in range(0, len(data), batch_size)]
    all_batch = []
    num_sample_all = 0
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
            num_sample_all += item_data["seq_len"] - 1
        all_batch.append(batch)

    # 数据集中标签和输入（历史正确率）之间相关性
    window_lens = [10, 20, 30, 40, 50]
    ppmcc = cal_PPMCC_label(all_batch, window_lens)
    print(ppmcc)
    for window_len in window_lens:
        for k in ["all", "label_eq0", "label_eq1"]:
            print(f"window length is {window_len}, ppmcc of {k:<5} is {ppmcc[window_len][k]:.4}")

