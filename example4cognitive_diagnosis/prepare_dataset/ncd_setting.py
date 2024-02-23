import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file, kt_data2cd_data
from lib.dataset.split_dataset import n_fold_split4CD_task2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009",
                        choices=("assist2009", "assist2009-full", "assist2012", "assist2017", "edi2020-task34",
                                 "edi2020-task1", "ednet-kt1", "xes3g5m", "algebra2005", "bridge2algebra2006",
                                 "statics2011", "slepemapy"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "ncd_setting"
    if params["dataset_name"] in ["assist2012", "assist2017", "edi2020-task34", "edi2020-task1", "statics2011", "slepemapy"]:
        params["data_type"] = "single_concept"
    else:
        params["data_type"] = "only_question"
    params["min_seq_len"] = 15
    params["n_fold"] = 5
    params["valid_radio"] = 0.1/0.8

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}
    params["lab_setting"] = {
        "name": params["setting_name"],
        "n_fold": params["n_fold"],
        "valid_radio": params["valid_radio"]
    }

    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)

    data = kt_data2cd_data(data_uniformed)
    n_fold_split4CD_task2(data, params, objects, min_seq_len=params["min_seq_len"])
