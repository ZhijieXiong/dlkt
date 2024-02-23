import json
import sys
import os
import inspect
import torch
import logging


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])

from lib.util.FileManager import FileManager
from lib.util.basic import *
from lib.util.data import write_json
from lib.util.parse import *
from lib.model.Module.KTEmbedLayer import KTEmbedLayer

from .util import config_optimizer


def general_config(local_params, global_params, global_objects):
    global_objects["logger"] = logging.getLogger("train_log")
    global_objects["logger"].setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(ch)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_params["save_model"] = local_params["save_model"]
    global_params["train_strategy"]["type"] = local_params["train_strategy"]
    global_params["device"] = "cuda" if (
            torch.cuda.is_available() and not local_params.get("use_cpu", False)
    ) else "cpu"
    if local_params.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)
    global_params["seed"] = local_params["seed"]

    # 训练策略配置
    num_epoch = local_params["num_epoch"]
    use_early_stop = local_params["use_early_stop"]
    epoch_early_stop = local_params["epoch_early_stop"]
    use_last_average = local_params["use_last_average"]
    epoch_last_average = local_params["epoch_last_average"]
    main_metric = local_params["main_metric"]
    use_multi_metrics = local_params["use_multi_metrics"]
    mutil_metrics = local_params["multi_metrics"]
    train_strategy_type = local_params["train_strategy"]

    global_params["train_strategy"] = {}
    train_strategy_config = global_params["train_strategy"]
    train_strategy_config["num_epoch"] = num_epoch
    train_strategy_config["main_metric"] = main_metric
    train_strategy_config["use_multi_metrics"] = use_multi_metrics
    if use_multi_metrics:
        train_strategy_config["multi_metrics"] = eval(mutil_metrics)

    train_strategy_config["type"] = train_strategy_type
    train_strategy_config[train_strategy_type] = {}
    if train_strategy_type == "valid_test":
        train_strategy_config["valid_test"]["use_early_stop"] = use_early_stop
        if use_early_stop:
            train_strategy_config["valid_test"]["epoch_early_stop"] = epoch_early_stop
    elif train_strategy_type == "no_valid":
        train_strategy_config["no_valid"]["use_average"] = use_last_average
        if use_last_average:
            train_strategy_config["no_valid"]["epoch_last_average"] = epoch_last_average
    else:
        raise NotImplementedError()

    # 数据集配置
    setting_name = local_params["setting_name"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]

    datasets_config = global_params["datasets_config"]
    datasets_config["train"]["setting_name"] = setting_name
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["valid"]["setting_name"] = setting_name
    if train_strategy_config["type"] == "valid_test":
        datasets_config["valid"]["file_name"] = valid_file_name
    datasets_config["train"]["file_name"] = train_file_name
    datasets_config["test"]["file_name"] = test_file_name

    global_objects["logger"].info(
        "basic setting\n"
        f"    device: {global_params['device']}, seed: {global_params['seed']}\n"
        "train policy\n"
        f"    type: {train_strategy_type}, {train_strategy_type}: {json.dumps(train_strategy_config[train_strategy_type])}, "
        f"train batch size: {local_params['train_batch_size']}, num of epoch: {num_epoch}\n"
        "evaluate metric\n"
        f"    main metric: {main_metric}, use multi metrics: {use_multi_metrics}{f', multi metrics: {mutil_metrics}' if use_multi_metrics else ''}"
    )

    # 优化器配置
    global_objects["logger"].info("optimizer setting")
    config_optimizer(local_params, global_params, global_objects, model_name="kt_model")

    # Q table，并解析Q table并得到相关数据
    dataset_name = local_params["dataset_name"]
    Q_table = file_manager.get_q_table(dataset_name, "only_question")
    if Q_table is None:
        Q_table = file_manager.get_q_table(dataset_name, "single_concept")
    global_objects["data"] = {}
    global_objects["data"]["Q_table"] = Q_table
    if Q_table is not None:
        # 如果有Q table的话，可以分析出question2concept_list和concept2question_list
        # 前者index表示习题id，value表示该习题对应的知识点列表
        # 后者index表示知识点id，value表示该知识点对应的习题列表
        global_objects["data"]["question2concept"] = question2concept_from_Q(global_objects["data"]["Q_table"])
        global_objects["data"]["concept2question"] = concept2question_from_Q(global_objects["data"]["Q_table"])
        # 如果有Q table的话，q2c_table和q2c_mask_table都是(num_q, num_max_c)的tensor
        # num_max_c表示在该数据集中一道习题最多对应几个知识点
        q2c_table, q2c_mask_table, num_max_concept = KTEmbedLayer.parse_Q_table(Q_table, global_params["device"])
        global_objects["data"]["q2c_table"] = q2c_table
        global_objects["data"]["q2c_mask_table"] = q2c_mask_table
        global_objects["data"]["num_max_concept"] = num_max_concept

    global_objects["logger"].info(
        "dataset\n"
        f"    setting: {setting_name}, dataset: {dataset_name}, train: {train_file_name}"
        f"{f', valid: {valid_file_name}' if train_strategy_type == 'valid_test' else ''}, test: {test_file_name}"
    )

    # 数据集统计信息
    statics_info_file_path = os.path.join(
        file_manager.get_setting_dir(setting_name),
        datasets_config["train"]["file_name"].replace(".txt", f"_statics.json")
    )
    if not os.path.exists(statics_info_file_path):
        global_objects["logger"].warning(
            f"\nWARNING: statics of train dataset (`{statics_info_file_path}`) is not exist. This file is required for some cases, e.g., "
            "fine grain evaluation such as long tail problem and some model using transfer_head2zero. "
            "If it is necessary, please run `prepare4fine_trained_evaluate.py` to generate statics of train dataset\n"
        )
    else:
        with open(statics_info_file_path, "r") as file:
            global_objects["data"]["train_data_statics"] = json.load(file)


def save_params(global_params, global_objects):
    file_manager = global_objects["file_manager"]
    model_root_dir = file_manager.get_models_dir()
    model_dir_name = global_params["save_model_dir_name"]
    model_dir = os.path.join(model_root_dir, model_dir_name)
    global_params["save_model_dir"] = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        assert False, f"{model_dir} exists"

    params_path = os.path.join(model_dir, "params.json")
    params_json = params2str(global_params)
    write_json(params_json, params_path)

    log_path = os.path.join(model_dir, "train_log.txt")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(fh)
