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
    transfer_head2zero = local_params.get("transfer_head2zero", False)
    head2tail_transfer_method = local_params.get("head2tail_transfer_method", "mean_pool")

    datasets_config = global_params["datasets_config"]
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["test"]["file_name"] = test_file_name
    datasets_config["test"]["kt"]["base_type"] = base_type

    # 细粒度配置
    fine_grain_config = global_params["evaluate"]["fine_grain"]
    fine_grain_config["max_seq_len"] = local_params["max_seq_len"]
    fine_grain_config["seq_len_absolute"] = eval(local_params["seq_len_absolute"])
    fine_grain_config["statics_path"] = local_params["statics_file_path"]

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    if data_type == "only_question":
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, "multi_concept")
    else:
        global_objects["data"]["Q_table"] = file_manager.get_q_table(dataset_name, data_type)

    # num_max_concept
    # preprocessed_dir = file_manager.get_preprocessed_dir(local_params["dataset_name"])
    # statics_preprocessed_multi_concept = load_json(os.path.join(preprocessed_dir,
    #                                                             "statics_preprocessed_multi_concept.json"))
    # global_params["num_max_concept"] = statics_preprocessed_multi_concept["num_max_concept"]

    # 是否将head的知识迁移到zero shot的知识
    global_params["transfer_head2zero"] = transfer_head2zero
    global_params["head2tail_transfer_method"] = head2tail_transfer_method

    return global_params, global_objects
