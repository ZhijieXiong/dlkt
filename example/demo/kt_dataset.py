import os.path

import torch.cuda
from torch.utils.data import DataLoader

from lib.util.FileManager import FileManager
from lib.dataset.KTDataset import KTDataset
from lib.dataset.util import parse_difficulty
from lib.util.data import read_preprocessed_file


if __name__ == "__main__":
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "datasets_config": {
            # 当前dataset的选择
            "dataset_this": "train",
            "data_type": "single_concept",
            "train": {
                # 两种数据格式，"kt" or "srs"，后者是序列推荐的格式
                "type": "kt4dimkt",
                "setting_name": "random_split_leave_multi_out_setting",
                "file_name": "assist2012_train_split_5.txt",
                "dataset_path": "",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    # 配置KTDataset需要的参数
                    "data_type": "multi_concept",
                    "base_type": "concept"
                },
                "kt4dimkt": {
                    # 配置DIMKT需要的参数
                    "num_min_question": 30,
                    "num_min_concept": 30,
                    "num_question_difficulty": 100,
                    "num_concept_difficulty": 100
                },
            },
            "valid": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_valid_fold_0.txt",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    "data_type": "multi_concept",
                    "base_type": "concept"
                },
            },
            "test": {
                "type": "kt",
                "setting_name": "pykt_setting",
                "file_name": "assist2009_test_fold_0.txt",
                "unuseful_seq_keys": {"user_id"},
                "kt": {
                    "data_type": "multi_concept",
                    "base_type": "concept"
                },
            }
        }
    }
    objects = {
        "file_manager": FileManager(r"F:\code\myProjects\dlkt"),
        "dimkt": {}
    }
    objects["dataset_this"] = read_preprocessed_file(os.path.join(
        objects["file_manager"].get_setting_dir(params["datasets_config"]["train"]["setting_name"]),
        params["datasets_config"]["train"]["file_name"]
    ))
    question_difficulty, concept_difficulty = \
        parse_difficulty(objects["dataset_this"],
                         params["datasets_config"]["data_type"],
                         params["datasets_config"]["train"]["kt4dimkt"]["num_min_question"],
                         params["datasets_config"]["train"]["kt4dimkt"]["num_min_concept"],
                         params["datasets_config"]["train"]["kt4dimkt"]["num_question_difficulty"],
                         params["datasets_config"]["train"]["kt4dimkt"]["num_concept_difficulty"])
    objects["dimkt"]["question_difficulty"] = question_difficulty
    objects["dimkt"]["concept_difficulty"] = concept_difficulty

    dataset = KTDataset(params, objects)
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        print("")
