import argparse
import os.path
import random

import numpy as np
from scipy.stats import norm
from scipy.stats import poisson

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type, get_keys_from_uniform, get_statics4lbkt
from lib.util.data import read_preprocessed_file, drop_qc, write2file
from lib.dataset.split_seq import dataset_truncate2multi_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009",
                        choices=("assist2009", "assist2012", "junyi2015", "assist2017"))
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
    if params["dataset_name"] == "junyi2015":
        # 只取长度最长的1000条序列
        seq_lens = list(map(lambda x: x["seq_len"], data_uniformed))
        max_indices = np.argpartition(np.array(seq_lens), -1000)[-1000:]
        data_uniformed_ = []
        for i in max_indices:
            data_uniformed_.append(data_uniformed[i])
        data_uniformed = data_uniformed_
    data_uniformed = drop_qc(data_uniformed, num2drop=10)

    if params["dataset_name"] != "assist2017":
        # 丢弃use_time_first小于0的数据（官方代码是如此处理的）
        id_keys, seq_keys = get_keys_from_uniform(data_uniformed)
        data_dropped = []
        num_dropped = 0
        for item_data in data_uniformed:
            item_data_new = {}
            for k in id_keys:
                item_data_new[k] = item_data[k]
            for k in seq_keys:
                item_data_new[k] = []
            for i in range(item_data["seq_len"]):
                use_time_first = item_data["use_time_first_seq"][i]
                if use_time_first <= 0:
                    num_dropped += 1
                    continue
                for k in seq_keys:
                    item_data_new[k].append(item_data[k][i])
            item_data_new["seq_len"] = len(item_data_new["question_seq"])
            if item_data_new["seq_len"] > 1:
                data_dropped.append(item_data_new)
        print(f"num of interaction dropped: {num_dropped}")
        data_uniformed = data_dropped

    # 生成time、hint和attempt factor
    use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict = \
        get_statics4lbkt(
            data_uniformed, use_time_first=params["dataset_name"] in ["assist2009", "assist2012", "junyi2015"]
        )
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

            if params["dataset_name"] == "assist2017":
                use_time_first = item_data["use_time_seq"][i]
            else:
                use_time_first = item_data["use_time_first_seq"][i]

            time_factor = 1 if (use_time_std == 0) else norm(use_time_mean, use_time_std).cdf(np.log(use_time_first))
            time_factor_seq.append(time_factor)

            num_attempt = item_data["num_attempt_seq"][i]
            attempt_factor = 1 - poisson(num_attempt_mean).cdf(num_attempt - 1)
            attempt_factor_seq.append(attempt_factor)

            num_hint = item_data["num_hint_seq"][i]
            hint_factor = 1 - poisson(num_hint_mean).cdf(num_hint - 1)
            hint_factor_seq.append(hint_factor)

            if (use_time_first <= 0) or (str(time_factor) == "nan"):
                print(f"time error: {use_time_first}, {time_factor}")
            if str(attempt_factor) == "nan":
                print(f"time error: {num_attempt}, {attempt_factor}")
            if str(hint_factor) == "nan":
                print(f"time error: {num_hint}, {hint_factor}")
        item_data["time_factor_seq"] = time_factor_seq
        item_data["attempt_factor_seq"] = attempt_factor_seq
        item_data["hint_factor_seq"] = hint_factor_seq

    # 改该论文的实验是先随机划分学生为训练集、验证集和测试集，然后再切割序列
    random.shuffle(data_uniformed)
    num_data_all = len(data_uniformed)
    n1 = int(num_data_all * 0.8)
    n2 = int(num_data_all * 0.9)
    data_train = data_uniformed[:n1]
    data_valid = data_uniformed[n1:n2]
    data_test = data_uniformed[n2:]

    dataset_train = dataset_truncate2multi_seq(data_train, params["min_seq_len"], params["max_seq_len"],
                                               single_concept=True)
    dataset_valid = dataset_truncate2multi_seq(data_valid, params["min_seq_len"], params["max_seq_len"],
                                               single_concept=True)
    dataset_test = dataset_truncate2multi_seq(data_test, params["min_seq_len"], params["max_seq_len"],
                                              single_concept=True)
    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])
    write2file(dataset_train, os.path.join(setting_dir, f"{params['dataset_name']}_train.txt"))
    write2file(dataset_valid, os.path.join(setting_dir, f"{params['dataset_name']}_valid.txt"))
    write2file(dataset_test, os.path.join(setting_dir, f"{params['dataset_name']}_test.txt"))
