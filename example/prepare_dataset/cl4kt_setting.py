import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type, str2bool
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2one_seq
from lib.dataset.split_dataset import n_fold_split2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009",
                        choices=("assist2009", "statics2011"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "cl4kt_setting"
    if params["dataset_name"] in ["statics2011"]:
        params["data_type"] = "only_question"
    else:
        params["data_type"] = "single_concept"
    params["max_seq_len"] = 50
    params["min_seq_len"] = 5
    params["n_fold"] = 5
    params["valid_radio"] = 0.1
    params["from_start"] = False

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
        "description": "序列处理：（1）序列长度小于100，则在后面补零；（2）序列长度大于100，则只取最后100次交互；\n"
                       "数据集划分：先用k折交叉划分为训练集和测试集，再在训练集中划分一部分数据为验证集",
        "data_type": params["data_type"],
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
                                                 multi_concept=params["data_type"] == "multi_concept",
                                                 from_start=params["from_start"])
    n_fold_split2(dataset_truncated, params, objects)
