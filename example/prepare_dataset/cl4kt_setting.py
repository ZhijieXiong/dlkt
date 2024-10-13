import argparse

import config

from lib.util.FileManager import FileManager
from lib.util.parse import parse_data_type
from lib.util.data import read_preprocessed_file
from lib.dataset.split_seq import dataset_truncate2one_seq
from lib.dataset.split_dataset import n_fold_split2
from lib.data_processor.util import process4CL4kt_assist2009


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2006", choices=("assist2009", "algebra2005", "algebra2006"))
    args = parser.parse_args()
    params = vars(args)

    params["setting_name"] = "cl4kt_setting"
    params["data_type"] = "single_concept"
    params["max_seq_len"] = 100
    params["min_seq_len"] = 5
    params["n_fold"] = 5
    params["valid_radio"] = 0.1
    params["from_start"] = False

    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["lab_setting"] = {
        "name": params["setting_name"],
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

    if params["dataset_name"] == "assist2009":
        concept_id2name = objects["file_manager"].get_concept_id2name("assist2009")
        concept_id_map = objects["file_manager"].get_concept_id_map("assist2009", "single_concept")
        data_uniformed = process4CL4kt_assist2009(data_uniformed, concept_id2name, concept_id_map)

    dataset_truncated = dataset_truncate2one_seq(data_uniformed,
                                                 params["min_seq_len"],
                                                 params["max_seq_len"],
                                                 single_concept=True,
                                                 from_start=params["from_start"])
    n_fold_split2(dataset_truncated, params, objects)
