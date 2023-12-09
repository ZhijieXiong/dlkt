import argparse

from lib.util.data import read_preprocessed_file, write_json
from lib.util.statics import dataset_basic_statics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_file_path", type=str,
                        help="用于从数据中提取信息，如每道习题出现的频率（长尾问题），准确率（偏差问题）",
                        default=r"F:\code\myProjects\dlkt\lab\settings\pykt_setting\assist2009_train_fold_0.txt")
    parser.add_argument("--data_type", type=str, default="multi_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    # 数据集信息
    parser.add_argument("--num_concept", type=int, default=149)
    parser.add_argument("--num_question", type=int, default=17751)
    # 划分知识点和习题频率为低中高所用的参数，用于研究长尾问题
    parser.add_argument("--concept_fre_low_middle", type=int, default=10)
    parser.add_argument("--concept_fre_middle_high", type=int, default=100)
    parser.add_argument("--question_fre_low_middle", type=int, default=5)
    parser.add_argument("--question_fre_middle_high", type=int, default=30)
    # 划分知识点和习题正确率为低中高所用的参数，用于研究偏差问题
    parser.add_argument("--concept_acc_low_middle", type=float, default=0.3)
    parser.add_argument("--concept_acc_middle_high", type=float, default=0.8)
    parser.add_argument("--question_acc_low_middle", type=float, default=0.3)
    parser.add_argument("--question_acc_middle_high", type=float, default=0.8)

    args = parser.parse_args()
    params = vars(args)

    data = read_preprocessed_file(params["target_file_path"])
    statics_info_file_path = params["target_file_path"].replace(".txt", f"_statics.json")
    basic_statics = dataset_basic_statics(data, params["data_type"],
                                          num_question=params["num_question"],
                                          num_concept=params["num_concept"])

    save_statics = {
        "acc_overall": basic_statics["acc_overall"],
        "question_low_fre": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: k_v_tuple[1] < params["question_fre_low_middle"],
                basic_statics["question_fre"].items()
            ))
        )),
        "question_middle_fre": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: params["question_fre_low_middle"] <= k_v_tuple[1] <= params["question_fre_middle_high"],
                basic_statics["question_fre"].items()
            ))
        )),
        "question_high_fre": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: k_v_tuple[1] > params["question_fre_middle_high"],
                basic_statics["question_fre"].items()
            ))
        )),
        "question_low_acc": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: k_v_tuple[1] < params["question_acc_low_middle"],
                basic_statics["question_acc"].items()
            ))
        )),
        "question_middle_acc": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: params["question_acc_low_middle"] <= k_v_tuple[1] <= params["question_acc_middle_high"],
                basic_statics["question_acc"].items()
            ))
        )),
        "question_high_acc": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: k_v_tuple[1] > params["question_acc_middle_high"],
                basic_statics["question_acc"].items()
            ))
        )),
    }
    if params["data_type"] != "only_ question":
        pass

