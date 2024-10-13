import os

import config

from lib.util.FileManager import FileManager
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2multi_seq
from lib.util.data import write2file


if __name__ == "__main__":
    params = {
        "setting_name": "our_setting_ood_by_country",
        "dataset_name": "slepemapy-anatomy",
        "data_type": "single_concept",
        "max_seq_len": 200,
        "min_seq_len": 5,
        "iid_radio": 0.2
    }
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}
    params["lab_setting"] = {
        "name": params["setting_name"],
        "description": "数据集划分：因为slepemapy这个数据集country 0占了80%，所以直接这个country作为训练集，属于所有数据作为测试集",
        "iid_radio": params["iid_radio"],
        "max_seq_len": params["max_seq_len"],
        "min_seq_len": params["min_seq_len"]
    }

    objects["file_manager"].add_new_setting(params["lab_setting"]["name"], params["lab_setting"])
    data_uniformed_path = objects["file_manager"].get_preprocessed_path(params["dataset_name"], params["data_type"])
    data_uniformed = read_preprocessed_file(data_uniformed_path)

    data_train = list(filter(lambda x: x["country_id"] == 0, data_uniformed))
    dataset_train_all = dataset_truncate2multi_seq(data_train,
                                                   params["min_seq_len"],
                                                   params["max_seq_len"],
                                                   single_concept=True)
    num_train = len(dataset_train_all)
    dataset_train = dataset_train_all[int(num_train * params["iid_radio"]):]
    dataset_test_iid = dataset_train_all[:int(num_train * params["iid_radio"])]
    data_test = list(filter(lambda x: x["country_id"] != 0, data_uniformed))
    dataset_test_ood = dataset_truncate2multi_seq(data_test,
                                                  params["min_seq_len"],
                                                  params["max_seq_len"],
                                                  single_concept=True)

    setting_dir = objects["file_manager"].get_setting_dir(params["setting_name"])
    train_path = os.path.join(setting_dir, f"{params['dataset_name']}_train_split_0.txt")
    test_iid_path = os.path.join(setting_dir, f"{params['dataset_name']}_valid_iid_split_0.txt")
    test_ood_path = os.path.join(setting_dir, f"{params['dataset_name']}_test_ood_split_0.txt")

    write2file(dataset_train, train_path)
    write2file(dataset_test_iid, test_iid_path)
    write2file(dataset_test_ood, test_ood_path)
