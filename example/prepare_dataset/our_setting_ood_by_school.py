import argparse
import os
import json

import config
import split_util

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.util.data import write2file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SLP-phy",
                        choices=("assist2009", "assist2012", "SLP-his", "SLP-phy"))
    parser.add_argument("--min_school_seq", type=int, default=70,
                        help="一所学校最少要有多少学生|序列"
                             "assist2009: 100, assist2012: 200, SLP: 70")
    parser.add_argument("--min_mean_seq_len", type=int, default=30)
    parser.add_argument("--train_test_radio_upper_bound", type=float, default=8.5/1.5)
    parser.add_argument("--train_test_radio_lower_bound", type=float, default=7/3)
    parser.add_argument("--iid_radio", type=float, default=0.2)
    parser.add_argument("--num_split", type=float, default=10, help="因为是随机划分，所以要设定划分多少组")
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "our_setting_ood_by_school"
    if params["dataset_name"] in ["assist2009"]:
        params["data_type"] = "only_question"
    else:
        params["data_type"] = "single_concept"
    params["max_seq_len"] = 200
    params["min_seq_len"] = 5
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "description": "数据集划分：先基于学校随机划分训练集和测试集，再在训练集内基于序列随机划分训练集和验证集，总共划分num_split组；"
                       "数据集划分具体步骤：（1）将小学校（序列数量少的学校，由min_school_seq定义）合并为一个学校，保证合并后每个学校"
                       "                    序列数量大于等于min_school_seq"
                       "                （2）根据train_test_radio_upper_bound和train_test_radio_lower_bound并基于学校划分"
                       "                    训练集和测试集，随机划分，循环进行，直到划分的训练集和测试机比例满足上面两个指标。并且保证"
                       "                    测试集不会出现学校序列平均长度很低（低于min_mean_seq_len），这种学校的数据不会出现在测试集"
                       "                （3）基于iid_radio和序列随机划分训练集为训练集和验证集"
                       "                （4）上述步骤循环进行，直到划分num_split组结果",
        "min_school_seq": params["min_school_seq"],
        "train_test_radio_upper_bound": params["train_test_radio_upper_bound"],
        "train_test_radio_lower_bound": params["train_test_radio_lower_bound"],
        "iid_radio": params["iid_radio"],
        "num_split": params["num_split"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"]
    }
    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])

    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)

    # 读取学校信息
    school_info_path = os.path.join(objects["file_manager"].get_preprocessed_dir(params["dataset_name"]),
                                    "school_info.json")
    with open(school_info_path, "r") as f:
        all_school_info = json.load(f)
    all_school_info = split_util.key_str2int(all_school_info)
    merged_schools, not_test_merged_schools = split_util.merge_school(all_school_info,
                                                                      min_school_user=params["min_school_seq"],
                                                                      min_mean_seq_len=params["min_mean_seq_len"])
    merged_schools = split_util.refine_merged_school_info(merged_schools, all_school_info)
    result_spilt_test_schools = []
    result_spilt_test_schools_str = []
    train_test_ratio_low = params["train_test_radio_lower_bound"]
    train_test_ratio_high = params["train_test_radio_upper_bound"]
    num_random = 0

    # 划分训练集和测试集
    while len(result_spilt_test_schools) < params["num_split"]:
        num_random += 1
        # 80%的学校做训练集
        schools_test, (total_train_num_user, total_test_num_user), (total_train_num_interaction, total_test_num_interaction) = (
            split_util.split(merged_schools, not_test_merged_schools, train_ratio=0.8))
        # 按照学校划分的训练集可能样本数过高或者过低，因为每个学校的序列数量不一样，需要做一次判断是否满足要求
        if train_test_ratio_low < (total_train_num_interaction / total_test_num_interaction) < train_test_ratio_high:
            schools_test_str = "-".join(list(map(str, sorted(schools_test))))
            if schools_test_str not in result_spilt_test_schools_str:
                result_spilt_test_schools.append(schools_test)
                result_spilt_test_schools_str.append(schools_test_str)

    print(num_random)
    print(result_spilt_test_schools)

    setting_dir = objects["file_manager"].get_setting_dir(params["setting_name"])
    for i, test_schools in enumerate(result_spilt_test_schools):
        data_train_iid, data_test_ood = split_util.split_data(data_uniformed, test_schools, merged_schools)
        dataset_train_iid = dataset_truncate2multi_seq(data_train_iid,
                                                       params["min_seq_len"],
                                                       params["max_seq_len"],
                                                       single_concept=True)
        num_train_iid = len(dataset_train_iid)
        num_test_iid = int(num_train_iid * params["iid_radio"])
        dataset_test_iid = dataset_train_iid[:num_test_iid]
        dataset_train = dataset_train_iid[num_test_iid:]
        dataset_test_ood = dataset_truncate2multi_seq(data_test_ood,
                                                      params["min_seq_len"],
                                                      params["max_seq_len"],
                                                      single_concept=True)

        train_path = os.path.join(setting_dir, f"{params['dataset_name']}_train_split_{i}.txt")
        test_iid_path = os.path.join(setting_dir, f"{params['dataset_name']}_valid_iid_split_{i}.txt")
        test_ood_path = os.path.join(setting_dir, f"{params['dataset_name']}_test_ood_split_{i}.txt")

        write2file(dataset_train, train_path)
        write2file(dataset_test_iid, test_iid_path)
        write2file(dataset_test_ood, test_ood_path)
