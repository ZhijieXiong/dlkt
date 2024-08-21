import os.path
import random

from copy import deepcopy

import config

from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file, write2file
from lib.dataset.split_seq import dataset_truncate2multi_seq

if __name__ == "__main__":
    params = {
        "setting_name": "llm4kt_setting",
        "dataset_name": "xes3g5m",
        "data_type": "only_question",
        "max_seq_len": 200,
        "min_seq_len": 2
    }

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}
    params["lab_setting"] = {
        "name": params["setting_name"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"],
    }
    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    data_uniformed = dataset_truncate2multi_seq(data_uniformed, params["min_seq_len"], params["max_seq_len"], single_concept=True)

    setting_dir = objects["file_manager"].get_setting_dir(params["lab_setting"]["name"])
    random.shuffle(data_uniformed)
    dataset_test = data_uniformed[:500]
    write2file(dataset_test, os.path.join(setting_dir, f"{params['dataset_name']}_test.txt"))

    nums_train = [1000, 2000, 3000, 4000, 5000, 10000]
    datasets_train = []
    for num_train in nums_train:
        datasets_train.append(deepcopy(data_uniformed[500:500 + num_train]))

    for num_train, dataset_train_valid in zip(nums_train, datasets_train):
        num_valid = int(num_train * 0.2)
        dataset_valid = dataset_train_valid[:num_valid]
        dataset_train = dataset_train_valid[num_valid:]
        write2file(dataset_valid, os.path.join(setting_dir, f"{params['dataset_name']}_valid_{num_train}.txt"))
        write2file(dataset_train, os.path.join(setting_dir, f"{params['dataset_name']}_train_{num_train}.txt"))

