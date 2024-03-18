import argparse
import os.path
import random

import numpy as np
from scipy.stats import norm
from scipy.stats import poisson

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type, get_keys_from_uniform
from lib.util.data import read_preprocessed_file, drop_qc, write2file
from lib.dataset.split_seq import dataset_truncate2multi_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2012", choices=("assist2009", "assist2012"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "lbkt_setting"
    if params["dataset_name"] == "assist2009":
        params["data_type"] = "only_question"
    else:
        params["data_type"] = "single_concept"
    params["max_seq_len"] = 100
    params["min_seq_len"] = 10

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"],
    }
    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])

    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    data_uniformed = drop_qc(data_uniformed, num2drop=10)

    # 丢弃use_time_first为0的数据（官方代码是如此处理的）
    id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
    data_dropped = []
    num_drop_interactions = 0
    for item_data in data_uniformed:
        item_data_new = {}
        for k in id_keys:
            item_data_new[k] = item_data[k]
        for k in seq_keys:
            item_data_new[k] = []
        for i in range(item_data["seq_len"]):
            use_time_first = item_data["use_time_first_seq"][i]
            if use_time_first == 0:
                num_drop_interactions += 1
                continue
            for k in seq_keys:
                item_data_new[k].append(item_data[k][i])
        item_data_new["seq_len"] = len(item_data_new["question_seq"])
        data_dropped.append(item_data_new)
    data_uniformed = data_dropped

    # 生成time、hint和attempt factor
    use_time_dict = {}
    num_attempt_dict = {}
    num_hint_dict = {}
    for item_data in data_uniformed:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            use_time_first = item_data["use_time_first_seq"][i]
            num_attempt = item_data["num_attempt_seq"][i]
            num_hint = item_data["num_hint_seq"][i]

            use_time_dict.setdefault(q_id, [])
            num_attempt_dict.setdefault(q_id, [])
            num_hint_dict.setdefault(q_id, [])

            use_time_dict[q_id].append(use_time_first)
            num_attempt_dict[q_id].append(num_attempt)
            num_hint_dict[q_id].append(num_hint)
    use_time_mean_dict = {k: np.mean(v) for k, v in use_time_dict.items()}
    use_time_std_dict = {k: np.var(v) for k, v in use_time_dict.items()}
    num_attempt_mean_dict = {k: np.mean(v) for k, v in num_attempt_dict.items()}
    num_hint_mean_dict = {k: np.mean(v) for k, v in num_hint_dict.items()}
    for item_data in data_uniformed:
        time_factor_seq = []
        attempt_factor_seq = []
        hint_factor_seq = []
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            use_time_mean = use_time_mean_dict[q_id]
            use_time_std = use_time_std_dict[q_id]
            num_attempt_mean = num_attempt_mean_dict[q_id]
            num_hint_mean = num_hint_mean_dict[q_id]

            use_time_first = item_data["use_time_first_seq"][i]
            time_factor = norm(use_time_mean, use_time_std).cdf(np.log(use_time_first))
            time_factor_seq.append(time_factor)

            num_attempt = item_data["num_attempt_seq"][i]
            attempt_factor = 1 - poisson(num_attempt_mean).cdf(num_attempt - 1)
            attempt_factor_seq.append(attempt_factor)

            num_hint = item_data["num_hint_seq"][i]
            hint_factor = 1 - poisson(num_hint_mean).cdf(num_hint - 1)
            hint_factor_seq.append(hint_factor)
        item_data["time_factor_seq"] = time_factor_seq
        del item_data["use_time_first_seq"]
        item_data["attempt_factor_seq"] = attempt_factor_seq
        del item_data["num_attempt_seq"]
        item_data["hint_factor_seq"] = hint_factor_seq
        del item_data["num_hint_seq"]

    # 改该论文的实验是先随机划分学生为训练集、验证集和测试集，然后再切割序列
    random.shuffle(data_uniformed)
    num_data_all = len(data_uniformed)
    n1 = int(num_data_all * 0.8)
    n2 = int(num_data_all * 0.9)
    data_train = data_uniformed[:n1]
    data_valid = data_uniformed[n1:n2]
    data_test = data_uniformed[n2:]

    dataset_train = dataset_truncate2multi_seq(data_train, params["min_seq_len"], params["max_seq_len"], single_concept=True)
    dataset_valid = dataset_truncate2multi_seq(data_valid, params["min_seq_len"], params["max_seq_len"], single_concept=True)
    dataset_test = dataset_truncate2multi_seq(data_test, params["min_seq_len"], params["max_seq_len"], single_concept=True)
    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])
    write2file(dataset_train, os.path.join(setting_dir, f"{params['dataset_name']}_train.txt"))
    write2file(dataset_valid, os.path.join(setting_dir, f"{params['dataset_name']}_valid.txt"))
    write2file(dataset_test, os.path.join(setting_dir, f"{params['dataset_name']}_test.txt"))
