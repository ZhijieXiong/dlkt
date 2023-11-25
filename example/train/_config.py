import json
import sys
import os
import inspect
import torch

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])

from lib.util.FileManager import FileManager
from lib.util.basic import *
from lib.util.data import write_json


def general_config(local_params, global_params, global_objects):
    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_params["save_model"] = local_params["save_model"]
    global_params["train_strategy"]["type"] = local_params["train_strategy"]
    global_params["datasets_config"]["data_type"] = local_params["data_type"]
    global_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # global_params["device"] = "cpu"

    # 训练策略配置
    num_epoch = local_params["num_epoch"]
    use_early_stop = local_params["use_early_stop"]
    epoch_early_stop = local_params["epoch_early_stop"]
    use_last_average = local_params["use_last_average"]
    epoch_last_average = local_params["epoch_last_average"]
    main_metric = local_params["main_metric"]
    use_multi_metrics = local_params["use_multi_metrics"]
    mutil_metrics = local_params["multi_metrics"]

    train_strategy = global_params["train_strategy"]
    train_strategy["num_epoch"] = num_epoch
    train_strategy["main_metric"] = main_metric
    train_strategy["use_multi_metrics"] = use_multi_metrics
    if use_multi_metrics:
        train_strategy["multi_metrics"] = eval(mutil_metrics)
    if train_strategy["type"] == "valid_test":
        train_strategy["valid_test"]["use_early_stop"] = use_early_stop
        if use_early_stop:
            train_strategy["valid_test"]["epoch_early_stop"] = epoch_early_stop
    else:
        train_strategy["no_valid"]["use_average"] = use_last_average
        if use_early_stop:
            train_strategy["no_valid"]["use_average"] = epoch_last_average

    # 数据集配置
    setting_name = local_params["setting_name"]
    data_type = local_params["data_type"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]

    datasets_config = global_params["datasets_config"]
    datasets_config["train"]["setting_name"] = setting_name
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["valid"]["setting_name"] = setting_name
    if train_strategy["type"] == "valid_test":
        datasets_config["valid"]["file_name"] = valid_file_name
    datasets_config["train"]["file_name"] = train_file_name
    datasets_config["test"]["file_name"] = test_file_name
    datasets_config["data_type"] = data_type

    # 优化器配置
    kt_optimizer_type = local_params["optimizer_type"]
    kt_weight_decay = local_params["weight_decay"]
    kt_momentum = local_params["momentum"]
    kt_learning_rate = local_params["learning_rate"]
    kt_enable_lr_schedule = local_params["enable_lr_schedule"]
    kt_lr_schedule_type = local_params["lr_schedule_type"]
    kt_lr_schedule_step = local_params["lr_schedule_step"]
    kt_lr_schedule_milestones = eval(local_params["lr_schedule_milestones"])
    kt_lr_schedule_gamma = local_params["lr_schedule_gamma"]
    kt_enable_clip_grad = local_params["enable_clip_grad"]
    kt_grad_clipped = local_params["grad_clipped"]

    optimizer_config = global_params["optimizers_config"]["kt_model"]
    optimizer_config["type"] = kt_optimizer_type
    optimizer_config[kt_optimizer_type]["lr"] = kt_learning_rate
    optimizer_config[kt_optimizer_type]["weight_decay"] = kt_weight_decay
    if kt_optimizer_type == "sgd":
        optimizer_config[kt_optimizer_type]["momentum"] = kt_momentum

    scheduler_config = global_params["schedulers_config"]["kt_model"]
    if kt_enable_lr_schedule:
        scheduler_config["use_scheduler"] = True
        scheduler_config["type"] = kt_lr_schedule_type
        if kt_lr_schedule_type == "StepLR":
            scheduler_config[kt_lr_schedule_type]["step_size"] = kt_lr_schedule_step
            scheduler_config[kt_lr_schedule_type]["gamma"] = kt_lr_schedule_gamma
        elif kt_lr_schedule_type == "MultiStepLR":
            scheduler_config[kt_lr_schedule_type]["milestones"] = kt_lr_schedule_milestones
            scheduler_config[kt_lr_schedule_type]["gamma"] = kt_lr_schedule_gamma
        else:
            raise NotImplementedError()
    else:
        scheduler_config["use_scheduler"] = False

    grad_clip_config = global_params["grad_clip_config"]["kt_model"]
    grad_clip_config["use_clip"] = kt_enable_clip_grad
    if kt_enable_clip_grad:
        grad_clip_config["grad_clipped"] = kt_grad_clipped


def save_params(global_params, global_objects):
    if global_params["save_model"]:
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
