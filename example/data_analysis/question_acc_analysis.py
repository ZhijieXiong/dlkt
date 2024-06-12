import argparse

import config

from lib.util.data import read_preprocessed_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\ednet-kt1\data_only_question.txt")
    args = parser.parse_args()
    params = vars(args)

    data = read_preprocessed_file(params["data_path"])
    num_acc_balance_seq = 0
    for item_data in data:
        acc = sum(item_data["correct_seq"][:item_data["seq_len"]]) / item_data["seq_len"]
        if 0.4 <= acc <= 0.6:
            num_acc_balance_seq += 1

    print(f"prop of balance acc seq is {num_acc_balance_seq / len(data)}")

