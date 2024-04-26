import argparse
import os
import random

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file, write2file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005",
                        choices=("assist2009", "assist2012", "algebra2005"))
    args = parser.parse_args()
    params = vars(args)

    setting_name = "ikt_setting"
    if params["dataset_name"] == "assist2012":
        data_type = "single_concept"
    else:
        data_type = "only_question"
    min_seq_len = 20
    n_fold = 5
    test_radio = 0.2
    seed = 0

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    params["lab_setting"] = {
        "name": setting_name,
        "min_seq_len": min_seq_len,
        "n_fold": n_fold,
        "test_radio": test_radio
    }

    file_manager.add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    setting_dir = file_manager.get_setting_dir(setting_name)
    parse_data_type(params["dataset_name"], data_type)
    data_uniformed_path = file_manager.get_preprocessed_path(params["dataset_name"], data_type)
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    data_uniformed = list(filter(lambda item: item["seq_len"] >= min_seq_len, data_uniformed))
    write2file(data_uniformed, os.path.join(setting_dir, f"{params['dataset_name']}_all.txt"))

    random.seed(seed)
    random.shuffle(data_uniformed)
    num_all = len(data_uniformed)
    num_fold = (num_all // n_fold) + 1
    dataset_folds = [data_uniformed[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    dataset_train = []
    dataset_test = []
    for i in range(n_fold):
        fold_valid = i
        dataset_test.append(dataset_folds[fold_valid])
        folds_train = set(range(n_fold)) - {fold_valid}
        data_train = []
        for fold in folds_train:
            data_train += dataset_folds[fold]
        dataset_train.append(data_train)

    for fold in range(n_fold):
        write2file(dataset_train[fold], os.path.join(setting_dir, f"{params['dataset_name']}_train_fold_{fold}.txt"))
        write2file(dataset_test[fold], os.path.join(setting_dir, f"{params['dataset_name']}_test_fold_{fold}.txt"))
