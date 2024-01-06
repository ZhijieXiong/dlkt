import os.path

import torch.cuda
from torch.utils.data import DataLoader

from lib.util.FileManager import FileManager
from lib.dataset.KTDataset_cpu2device import KTDataset_cpu2device
from lib.dataset.util import parse4dataset_enhanced, parse_low_fre_question
from lib.util.data import read_preprocessed_file
from lib.util.parse import question2concept_from_Q, concept2question_from_Q


if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "datasets_config": {
            # 当前dataset的选择
            "dataset_this": "train",
            "data_type": "single_concept",
            "train": {
                "type": "kt_output_enhance",
                "setting_name": "random_split_leave_multi_out_setting",
                "file_name": "assist2012_train_split_5.txt",
                "dataset_path": "",
                "unuseful_seq_keys": {"user_id"},
                "kt_enhance": {
                    "num_min_question4diff": 30,
                    "hard_acc": 0.4,
                    "easy_acc": 0.8
                },
            },
        }
    }
    objects = {
        "file_manager": FileManager(r"F:\code\myProjects\dlkt")
    }
    objects["dataset_this"] = read_preprocessed_file(os.path.join(
        objects["file_manager"].get_setting_dir(params["datasets_config"]["train"]["setting_name"]),
        params["datasets_config"]["train"]["file_name"]
    ))
    Q_table = FileManager(r"F:\code\myProjects\dlkt").get_q_table("assist2012", "single_concept")
    question2concept = question2concept_from_Q(Q_table)
    concept2question = concept2question_from_Q(Q_table)

    parse_low_fre_question(
        objects["dataset_this"],
        params["datasets_config"]["data_type"],
        5,
        53091
    )

    concept_dict, question_dict = parse4dataset_enhanced(objects["dataset_this"],
                                                         params["datasets_config"]["data_type"],
                                                         params["datasets_config"]["train"]["kt_enhance"]["num_min_question4diff"],
                                                         question2concept,
                                                         concept2question,
                                                         params["datasets_config"]["train"]["kt_enhance"]["hard_acc"],
                                                         params["datasets_config"]["train"]["kt_enhance"]["easy_acc"])

    objects["data"] = {}
    objects["data"]["Q_table"] = Q_table
    objects["data"]["question2concept"] = question2concept
    objects["data"]["concept2question"] = concept2question
    objects["kt_enhance"] = {}
    objects["kt_enhance"]["concept_dict"] = concept_dict
    objects["kt_enhance"]["question_dict"] = question_dict

    dataset = KTDataset_cpu2device(params, objects)
    dataloader = DataLoader(dataset, batch_size=64)

    for batch in dataloader:
        print("")
