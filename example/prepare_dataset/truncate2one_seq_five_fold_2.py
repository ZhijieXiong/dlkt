import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type, str2bool
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2one_seq
from lib.dataset.split_dataset import n_fold_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    # setting config
    parser.add_argument("--setting_name", type=str, default="multi_concept-truncate2one_seq_five_fold_2")
    parser.add_argument("--data_type", type=str, default="multi_concept",
                        choices=("multi_concept", "single_concept", "only_question"))
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--valid_radio", type=float, default=0.2)
    parser.add_argument("--from_start", type=str2bool, default=True)

    args = parser.parse_args()
    params = vars(args)
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "description": "序列处理：（1）序列长度小于n，则在后面补零；（2）序列长度大于n，则截断成多条序列；\n"
                       "数据集划分：先用k折交叉划分为训练集和测试集，再在训练集中划分一部分数据为验证集",
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"],
        "n_fold": params["n_fold"],
        "valid_radio": params["valid_radio"],
        "from_start": params["from_start"]
    }
    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])

    parse_data_type(params["dataset_name"], params["data_type"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)
    dataset_truncated = dataset_truncate2one_seq(data_uniformed,
                                                 params["min_seq_len"],
                                                 params["max_seq_len"],
                                                 single_concept=params["data_type"] != "multi_concept",
                                                 from_start=params["from_start"])
    n_fold_split2(dataset_truncated, params, objects)
