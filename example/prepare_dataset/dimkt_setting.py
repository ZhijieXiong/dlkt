import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file, drop_qc
from lib.data_processor.util import process4DIMKT
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.dataset.split_dataset import n_fold_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2012", choices=("assist2012", "edi2020-task1"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "dimkt_setting"
    params["data_type"] = "single_concept"
    params["max_seq_len"] = 100
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
    if params["dataset_name"] == "edi2020-task1":
        # 只使用该比赛的训练集
        data_uniformed = list(filter(lambda x: x["dataset_type"] == 0, data_uniformed))
    data_uniformed = drop_qc(data_uniformed, num2drop=30)
    data_uniformed = process4DIMKT(data_uniformed, num_q_level=100, num_c_level=100)
    dataset_truncated = dataset_truncate2multi_seq(data_uniformed,
                                                   params["min_seq_len"],
                                                   params["max_seq_len"],
                                                   single_concept=True)
    n_fold_split2(dataset_truncated, params, objects)
