import argparse

import config

from lib.util.FileManager import FileManager
from lib.data_processor.DataProcessor import DataProcessor
from lib.util.data import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xes3g5m",
                        choices=("assist2009", "assist2009-full", "assist2012", "assist2015", "assist2017",
                                 "algebra2005", "algebra2006", "algebra2008",
                                 "bridge2algebra2006", "bridge2algebra2008",
                                 "edi2020-task1", "edi2020-task34",
                                 "SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy",
                                 "ednet-kt1", "slepemapy-anatomy", "xes3g5m", "statics2011", "junyi2015", "poj"))

    args = parser.parse_args()
    params = vars(args)
    objects = {"file_manager": FileManager(config.FILE_MANAGER_ROOT)}

    params["preprocess_config"] = {
        "dataset_name": params["dataset_name"],
        "data_path": objects["file_manager"].get_dataset_raw_path(params["dataset_name"])
    }

    print(f"processing {params['dataset_name']} ...")
    data_processor = DataProcessor(params, objects)
    data_uniformed = data_processor.process_data()
    Q_table = data_processor.Q_table
    data_statics_raw = data_processor.statics_raw
    data_statics_preprocessed = data_processor.statics_preprocessed

    print(f"saving data of {params['dataset_name']} ...")
    objects["file_manager"].save_data_statics_raw(data_statics_raw, params["dataset_name"])
    dataset_name = params["dataset_name"]
    for k in data_uniformed:
        data_path = objects["file_manager"].get_preprocessed_path(dataset_name, k)
        write2file(data_uniformed[k], data_path)
        if k == "only_question":
            if dataset_name in ["assist2015", "poj"]:
                objects["file_manager"].save_data_statics_processed(data_statics_preprocessed[k], dataset_name, k)
            continue
        objects["file_manager"].save_data_statics_processed(data_statics_preprocessed[k], dataset_name, k)
        objects["file_manager"].save_q_table(Q_table[k], dataset_name, k)

    all_id_maps = data_processor.get_all_id_maps()
    objects["file_manager"].save_data_id_map(all_id_maps, dataset_name)
    print(f"finsh processing and saving successfully")
