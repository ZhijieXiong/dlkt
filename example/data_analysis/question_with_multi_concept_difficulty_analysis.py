import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file
from lib.util.parse import question2concept_from_Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009",
                        choices=("assist2009", "assist2009-new", "ednet-kt1", "xes3g5m"))
    parser.add_argument("--data_path", type=str,
                        default=r"F:\code\myProjects\dlkt\lab\dataset_preprocessed\ednet-kt1\data_only_question.txt")
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    Q_table = file_manager.get_q_table(params["dataset_name"], "only_question")
    q2c_table = question2concept_from_Q(Q_table)
    data = read_preprocessed_file(params["data_path"])

    # 按照知识点数目对习题分类，统计正确率
    q_acc_num_c = {}
    question_acc = {}
    for item_data in data:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            correct = item_data["correct_seq"][i]
            num_c = len(q2c_table[q_id])
            question_acc.setdefault(num_c, {"right": 0, "count": 0})
            question_acc[num_c]["right"] += correct
            question_acc[num_c]["count"] += 1

    for k, v in question_acc.items():
        right = v["right"]
        count = v["count"]
        q_acc_num_c[k] = right / count

    num_c_sorted = sorted(list(q_acc_num_c.keys()))
    for k in num_c_sorted:
        print(f"num sample and acc of question with {k} concepts: {question_acc[k]['count']}, {q_acc_num_c[k]}")
