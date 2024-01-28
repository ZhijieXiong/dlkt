import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.dataset.split_dataset import n_fold_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="statics2011",
                        choices=("assist2009", "assist2012", "assist2017", "edi2020-task34", "ednet-kt1", "xes3g5m",
                                 "algebra2005", "bridge2algebra2006", "statics2011"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "our_setting"
    if params["dataset_name"] in ["assist2012", "assist2017", "edi2020-task34", "statics2011"]:
        params["data_type"] = "single_concept"
    else:
        params["data_type"] = "only_question"
    params["max_seq_len"] = 200
    params["min_seq_len"] = 10
    params["n_fold"] = 5
    params["valid_radio"] = 0.2

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}
    params["lab_setting"] = {
        "name": params["setting_name"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"],
        "n_fold": params["n_fold"],
        "valid_radio": params["valid_radio"]
    }

    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    dataset_truncated = dataset_truncate2multi_seq(data_uniformed,
                                                   params["min_seq_len"],
                                                   params["max_seq_len"],
                                                   single_concept=True)
    n_fold_split2(dataset_truncated, params, objects)
