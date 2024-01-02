import argparse
import json

import config

from lib.util.data import read_preprocessed_file
from lib.util.statics import dataset_basic_statics
from lib.util.parse import str2bool


def sort_dict(d):
    d_list = list(d.items())
    # 过滤掉-1主要是针对acc，因为前面处理中会把训练集中没出现过的习题或者知识点的正确率设为-1
    d_list = list(filter(lambda x: x[1] != -1, d_list))
    d_list = sorted(d_list, key=lambda x: x[1])
    d_list = list(map(lambda x: x[0], d_list))

    return d_list


def extract_subset(d, num1_extract, num2_extract, is_acc=False):
    """
    从一个d中抽取子集，如果num1_extract是None，表示提取最高的num2_extract（如果num2_extract是整数，表示绝对值筛选，如果是小数，表示相对值筛选）；
    反之选最低的num1_extract；如果两个值都不是None，表示取中间的
    :param d: dict
    :param num1_extract: int / float / None, if int or float and num2_extract is not None, num1_extract must be same type to num2_extract
    :param num2_extract:int / float / None, if int or float and num1_extract is not None, num2_extract must be same type to num1_extract
    :param is_acc: 如果是acc的话，整数要转换为小数，即除100
    :return:
    """
    if num2_extract is None:
        if type(num1_extract) is int:
            if is_acc:
                num1_extract /= 100
            return list(map(
                lambda k_v_tuple: k_v_tuple[0],
                list(filter(
                    lambda k_v_tuple: k_v_tuple[1] < num1_extract,
                    d.items()
                ))
            ))
        else:
            d_list = sort_dict(d)
            num_all = len(d_list)
            return d_list[:int(num1_extract * num_all)]
    elif num1_extract is None:
        if type(num2_extract) is int:
            if is_acc:
                num2_extract /= 100
            return list(map(
                lambda k_v_tuple: k_v_tuple[0],
                list(filter(
                    lambda k_v_tuple: k_v_tuple[1] > num2_extract,
                    d.items()
                ))
            ))
        else:
            d_list = sort_dict(d)
            num_all = len(d_list)
            return d_list[int(num2_extract * num_all):]
    else:
        if type(num1_extract) is int:
            if is_acc:
                num1_extract /= 100
                num2_extract /= 100
            return list(map(
                lambda k_v_tuple: k_v_tuple[0],
                list(filter(
                    lambda k_v_tuple: num1_extract < k_v_tuple[1] < num2_extract,
                    d.items()
                ))
            ))
        else:
            d_list = sort_dict(d)
            num_all = len(d_list)
            return d_list[int(num1_extract * num_all): int(num2_extract * num_all)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_file_path", type=str,
                        help="用于从数据中提取信息，如每道习题出现的频率（长尾问题），准确率（偏差问题）",
                        default=r"F:\code\myProjects\dlkt\lab\settings\random_split_leave_multi_out_setting\assist2012_train_split_5.txt")
    parser.add_argument("--data_type", type=str, default="single_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    # 数据集信息
    parser.add_argument("--num_concept", type=int, default=265)
    parser.add_argument("--num_question", type=int, default=53091)
    # 划分知识点和习题频率为低中高所用的参数，用于研究长尾问题
    parser.add_argument("--use_absolute4fre", type=str2bool, default=False)
    parser.add_argument("--concept_fre_low_middle", type=int, default=100)
    parser.add_argument("--concept_fre_middle_high", type=int, default=1000)
    parser.add_argument("--question_fre_low_middle", type=int, default=5)
    parser.add_argument("--question_fre_middle_high", type=int, default=20)
    parser.add_argument("--question_fre_percent_lowest", type=float, default=0.3)
    parser.add_argument("--question_fre_percent_highest", type=float, default=0.8)
    parser.add_argument("--concept_fre_percent_lowest", type=float, default=0.5)
    parser.add_argument("--concept_fre_percent_highest", type=float, default=0.8)
    # 划分知识点和习题正确率为低中高所用的参数，用于研究偏差问题
    parser.add_argument("--use_absolute4acc", type=str2bool, default=True)
    parser.add_argument("--concept_acc_low_middle", type=int, default=20, help="单位%")
    parser.add_argument("--concept_acc_middle_high", type=int, default=80, help="单位%")
    parser.add_argument("--question_acc_low_middle", type=int, default=20, help="单位%")
    parser.add_argument("--question_acc_middle_high", type=int, default=80, help="单位%")
    parser.add_argument("--question_acc_percent_lowest", type=float, default=0.2)
    parser.add_argument("--question_acc_percent_highest", type=float, default=0.8)
    parser.add_argument("--concept_acc_percent_lowest", type=float, default=0.2)
    parser.add_argument("--concept_acc_percent_highest", type=float, default=0.8)

    args = parser.parse_args()
    params = vars(args)

    data = read_preprocessed_file(params["target_file_path"])
    statics_info_file_path = params["target_file_path"].replace(".txt", f"_statics.json")
    basic_statics = dataset_basic_statics(data, params["data_type"],
                                          num_question=params["num_question"],
                                          num_concept=params["num_concept"])
    if params["use_absolute4fre"]:
        question_fre_low_middle = params["question_fre_low_middle"]
        question_fre_middle_high = params["question_fre_middle_high"]
        concept_fre_low_middle = params["concept_fre_low_middle"]
        concept_fre_middle_high = params["concept_fre_middle_high"]
    else:
        question_fre_low_middle = params["question_fre_percent_lowest"]
        question_fre_middle_high = params["question_fre_percent_highest"]
        concept_fre_low_middle = params["concept_fre_percent_lowest"]
        concept_fre_middle_high = params["concept_fre_percent_highest"]
    if params["use_absolute4acc"]:
        question_acc_low_middle = params["question_acc_low_middle"]
        question_acc_middle_high = params["question_acc_middle_high"]
        concept_acc_low_middle = params["concept_acc_low_middle"]
        concept_acc_middle_high = params["concept_acc_middle_high"]
    else:
        question_acc_low_middle = params["question_acc_percent_lowest"]
        question_acc_middle_high = params["question_acc_percent_highest"]
        concept_acc_low_middle = params["concept_acc_percent_lowest"]
        concept_acc_middle_high = params["concept_acc_percent_highest"]

    save_statics = {
        "acc_overall": basic_statics["acc_overall"],
        "question_low_fre": extract_subset(basic_statics["question_fre"],
                                           question_fre_low_middle,
                                           None, False),
        "question_middle_fre": extract_subset(basic_statics["question_fre"],
                                              question_fre_low_middle,
                                              question_fre_middle_high, False),
        "question_high_fre": extract_subset(basic_statics["question_fre"],
                                            None,
                                            question_fre_middle_high, False),
        "question_low_acc": extract_subset(basic_statics["question_acc"],
                                           question_acc_low_middle,
                                           None, True),
        "question_middle_acc": extract_subset(basic_statics["question_acc"],
                                              question_acc_low_middle,
                                              question_acc_middle_high, True),
        "question_high_acc": extract_subset(basic_statics["question_acc"],
                                            None,
                                            question_acc_middle_high, True),
        "question_zero_fre": list(map(
            lambda k_v_tuple: k_v_tuple[0],
            list(filter(
                lambda k_v_tuple: k_v_tuple[1] == 0,
                basic_statics["question_fre"].items()
            ))
        ))
    }
    save_statics['question_low_fre'] = list(set(save_statics['question_low_fre']) - set(save_statics['question_zero_fre']))

    # 知识点
    if params["data_type"] != "only_question":
        save_statics["concept_low_fre"] = extract_subset(basic_statics["concept_fre"],
                                                         concept_fre_low_middle,
                                                         None, False)
        save_statics["concept_middle_fre"] = extract_subset(basic_statics["concept_fre"],
                                                            concept_fre_low_middle,
                                                            concept_fre_middle_high, False)
        save_statics["concept_high_fre"] = extract_subset(basic_statics["concept_fre"],
                                                          None,
                                                          concept_fre_middle_high, False)
        save_statics["concept_low_acc"] = extract_subset(basic_statics["concept_acc"],
                                                         concept_acc_low_middle,
                                                         None, True)
        save_statics["concept_middle_acc"] = extract_subset(basic_statics["concept_acc"],
                                                            concept_acc_low_middle,
                                                            concept_acc_middle_high, True)
        save_statics["concept_high_acc"] = extract_subset(basic_statics["concept_acc"],
                                                          None,
                                                          concept_acc_middle_high, True)

    with open(statics_info_file_path, "w") as f:
        json.dump(save_statics, f)
