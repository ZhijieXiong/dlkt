import os.path

import torch
import logging
import sys

from config import FILE_MANAGER_ROOT

from lib.util.FileManager import FileManager
from lib.util.basic import get_now_time


def evaluate_general_config(local_params):
    global_params = {}
    global_objects = {}

    # 配置文件管理器和日志
    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_objects["logger"] = logging.getLogger("evaluate_log")
    global_objects["logger"].setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(ch)
    log_path = os.path.join(local_params["save_model_dir"], f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(fh)

    save_model_dir = local_params["save_model_dir"]
    setting_name = local_params["setting_name"]
    test_file_name = local_params["test_file_name"]
    base_type = local_params["base_type"]
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    statics_file_path = local_params["statics_file_path"]
    max_seq_len = local_params["max_seq_len"]
    seq_len_absolute = local_params["seq_len_absolute"]
    transfer_head2zero = local_params.get("transfer_head2zero", False)
    head2tail_transfer_method = local_params.get("head2tail_transfer_method", "mean_pool")

    global_params["save_model_dir"] = save_model_dir
    global_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试数据集配置
    global_params["datasets_config"] = {
        "dataset_this": "test",
        "data_type": data_type,
        "test": {
            "type": "kt",
            "setting_name": setting_name,
            "file_name": test_file_name,
            "unuseful_seq_keys": {"user_id"},
            "kt": {
                "base_type": base_type
            },
        }
    }

    # 细粒度配置
    global_params["evaluate"] = {"fine_grain": {}}
    fine_grain_config = global_params["evaluate"]["fine_grain"]
    fine_grain_config["max_seq_len"] = max_seq_len
    fine_grain_config["seq_len_absolute"] = eval(seq_len_absolute)
    fine_grain_config["statics_path"] = statics_file_path

    # Q_table
    global_objects["data"] = {}
    if data_type == "only_question":
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, "multi_concept")
    else:
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, data_type)

    # 是否将head的知识迁移到zero shot的知识
    global_params["transfer_head2zero"] = {}
    global_params["transfer_head2zero"]["use_transfer"] = transfer_head2zero
    global_params["transfer_head2zero"]["transfer_method"] = head2tail_transfer_method

    return global_params, global_objects
