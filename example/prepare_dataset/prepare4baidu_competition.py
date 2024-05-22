import os

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file, write2file
from lib.dataset.split_seq import dataset_truncate2multi_seq


if __name__ == "__main__":
    dataset_name = "xes3g5m"
    setting_name = "baidu_competition"
    data_type = "only_question"
    max_seq_len = 200
    min_seq_len = 3

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    lab_setting = {
        "name": setting_name,
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len
    }

    file_manager.add_new_setting(setting_name, lab_setting)
    parse_data_type(dataset_name, data_type)
    data_uniformed_path = file_manager.get_preprocessed_path(dataset_name, data_type)
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    num = len(data_uniformed)
    num_train_valid = int(num * 0.9)
    num_train = int(num_train_valid * 0.8)
    data_train = data_uniformed[:num_train]
    data_valid = data_uniformed[num_train:num_train_valid]
    data_test = data_uniformed[num_train_valid:]
    dataset_train = dataset_truncate2multi_seq(data_train, min_seq_len, max_seq_len, single_concept=True)
    dataset_valid = dataset_truncate2multi_seq(data_valid, min_seq_len, max_seq_len, single_concept=True)
    dataset_test = dataset_truncate2multi_seq(data_test, min_seq_len, max_seq_len, single_concept=True)

    setting_dir = file_manager.get_setting_dir(setting_name)
    write2file(dataset_train, os.path.join(setting_dir, "train_dataset.txt"))
    write2file(dataset_valid, os.path.join(setting_dir, "valid_dataset.txt"))
    write2file(dataset_test, os.path.join(setting_dir, "test_dataset.txt"))
    write2file(data_test, os.path.join(setting_dir, "test_data.txt"))
