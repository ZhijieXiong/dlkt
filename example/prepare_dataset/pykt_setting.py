import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.dataset.split_dataset import n_fold_split1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2017")
    # setting config
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--test_radio", type=float, default=0.2)
    parser.add_argument("--valid_radio", type=float, default=0.2)

    args = parser.parse_args()
    params = vars(args)
    if params["dataset_name"] in ["assist2009", "assist2009-new", "xes3g5m"]:
        params["data_type"] = "multi_concept"
    elif params["dataset_name"] in ["assist2015", "statics2011"]:
        params["data_type"] = "only_question"
    else:
        params["data_type"] = "single_concept"
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "description": "序列处理：（1）序列长度小于200，则在后面补零；（2）序列长度大于200，则截断成多条序列；\n"
                       "数据集划分：选一部分数据做测试集，剩余数据用k折交叉划分为训练集和验证集",
        "data_type": params["data_type"],
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
                                                   single_concept=params["data_type"] != "multi_concept")
    n_fold_split1(dataset_truncated, params, objects)
