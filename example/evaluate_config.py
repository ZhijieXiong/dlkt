import os.path

import torch
import logging
import sys

from config import FILE_MANAGER_ROOT

from lib.util.parse import question2concept_from_Q, concept2question_from_Q
from lib.model.Module.KTEmbedLayer import KTEmbedLayer
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

    # gpu或者cpu配置
    global_params["device"] = "cuda" if (
            torch.cuda.is_available() and not local_params.get("use_cpu", False)
    ) else "cpu"
    if local_params.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)

    # 加载模型参数配置
    save_model_dir = local_params["save_model_dir"]
    global_params["save_model_dir"] = save_model_dir

    # 测试配置
    setting_name = local_params["setting_name"]
    test_file_name = local_params["test_file_name"]
    base_type = local_params["base_type"]
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    train_statics_common_path = local_params["train_statics_common_path"]
    train_statics_special_path = local_params["train_statics_special_path"]

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

    # 细粒度测试配置
    max_seq_len = local_params["max_seq_len"]
    seq_len_absolute = local_params["seq_len_absolute"]

    global_params["evaluate"] = {"fine_grain": {}}
    fine_grain_config = global_params["evaluate"]["fine_grain"]
    fine_grain_config["max_seq_len"] = max_seq_len
    fine_grain_config["seq_len_absolute"] = eval(seq_len_absolute)
    fine_grain_config["train_statics_common_path"] = train_statics_common_path
    fine_grain_config["train_statics_special_path"] = train_statics_special_path

    # Q_table
    global_objects["data"] = {}
    if data_type == "only_question":
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, "multi_concept")
    else:
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, data_type)
    Q_table = global_objects["data"]["Q_table"]
    if Q_table is not None:
        global_objects["data"]["Q_table_tensor"] = \
            torch.from_numpy(global_objects["data"]["Q_table"]).long().to(global_params["device"])
        global_objects["data"]["question2concept"] = question2concept_from_Q(Q_table)
        global_objects["data"]["concept2question"] = concept2question_from_Q(Q_table)
        q2c_table, q2c_mask_table, num_max_concept = KTEmbedLayer.parse_Q_table(Q_table, global_params["device"])
        global_objects["data"]["q2c_table"] = q2c_table
        global_objects["data"]["q2c_mask_table"] = q2c_mask_table
        global_objects["data"]["num_max_concept"] = num_max_concept

    return global_params, global_objects
