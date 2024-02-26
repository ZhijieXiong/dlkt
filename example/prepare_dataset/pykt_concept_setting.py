import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.dataset.split_dataset import n_fold_split1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m",
                        choices=("assist2009", "algebra2005", "bridge2algebra2006", "xes3g5m"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "pykt_concept_setting"
    # 对多知识点习题，将其展开为多个记录，测试的时候再聚合回去
    params["data_type"] = "multi_concept"
    params["max_seq_len"] = 200
    params["min_seq_len"] = 3
    params["n_fold"] = 5
    params["test_radio"] = 0.2
    params["valid_radio"] = 0.2

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}
    params["lab_setting"] = {
        "name": params["setting_name"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"],
        "n_fold": params["n_fold"],
        "test_radio": params["test_radio"],
        "valid_radio": params["valid_radio"]
    }

    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    dataset_truncated = dataset_truncate2multi_seq(data_uniformed,
                                                   params["min_seq_len"],
                                                   params["max_seq_len"],
                                                   single_concept=False)
    n_fold_split1(dataset_truncated, params, objects)
