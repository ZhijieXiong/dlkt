import argparse
import numpy as np

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.dataset.split_dataset import n_fold_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="moocradar-C_746997",
                        choices=("assist2009", "assist2012", "assist2017", "edi2020-task1", "edi2020-task34",
                                 "ednet-kt1", "xes3g5m", "statics2011", "slepemapy-anatomy", "junyi2015",
                                 "moocradar-C_746997"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "our_setting"
    if params["dataset_name"] in ["assist2012", "assist2017", "edi2020-task34", "edi2020-task1", "statics2011",
                                  "slepemapy", "junyi2015", "SLP-his", "SLP-phy"]:
        params["data_type"] = "single_concept"
    else:
        params["data_type"] = "only_question"
    params["max_seq_len"] = 200
    params["min_seq_len"] = 3
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

    if params["dataset_name"] == "junyi2015":
        # 只取长度最长的1000条序列
        seq_lens = list(map(lambda x: x["seq_len"], data_uniformed))
        max_indices = np.argpartition(np.array(seq_lens), -1000)[-1000:]
        data_uniformed_ = []
        for i in max_indices:
            data_uniformed_.append(data_uniformed[i])
        data_uniformed = data_uniformed_

    dataset_truncated = dataset_truncate2multi_seq(data_uniformed,
                                                   params["min_seq_len"],
                                                   params["max_seq_len"],
                                                   single_concept=True)
    n_fold_split2(dataset_truncated, params, objects)
