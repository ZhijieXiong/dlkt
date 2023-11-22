import torch
from copy import deepcopy

from config import FILE_MANAGER_ROOT

from lib.template.evaluate_params_template import EVALUATE_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.FileManager import FileManager


def evaluate_general_config(local_params):
    global_params = deepcopy(EVALUATE_PARAMS)
    global_objects = deepcopy(OBJECTS)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_params["save_model_dir"] = local_params["save_model_dir"]
    global_params["datasets_config"]["data_type"] = local_params["data_type"]
    global_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试数据集配置
    setting_name = local_params["setting_name"]
    test_file_name = local_params["test_file_name"]
    base_type = local_params["base_type"]

    datasets_config = global_params["datasets_config"]
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["test"]["file_name"] = test_file_name
    datasets_config["test"]["kt"]["base_type"] = base_type

    # 细粒度配置
    evaluate_config = global_params["evaluate"]["fine_grain"]
    evaluate_config["max_seq_len"] = local_params["max_seq_len"]
    evaluate_config["seq_len_absolute"] = eval(local_params["seq_len_absolute"])
    evaluate_config["seq_len_percent"] = eval(local_params["seq_len_percent"])

    return global_params, global_objects
