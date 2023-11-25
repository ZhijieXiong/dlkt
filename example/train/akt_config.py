from copy import deepcopy
from _config import *
from _cl_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def akt_general_config(local_params, global_params):
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "AKT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    key_query_same = local_params["key_query_same"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    separate_qa = local_params["separate_qa"]
    dropout = local_params["dropout"]

    # encoder layer
    akt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    akt_encoder_layer_config["num_concept"] = num_concept
    akt_encoder_layer_config["num_question"] = num_question
    akt_encoder_layer_config["dim_model"] = dim_model
    akt_encoder_layer_config["key_query_same"] = key_query_same
    akt_encoder_layer_config["num_head"] = num_head
    akt_encoder_layer_config["num_block"] = num_block
    akt_encoder_layer_config["dim_ff"] = dim_ff
    akt_encoder_layer_config["dim_final_fc"] = dim_final_fc
    akt_encoder_layer_config["separate_qa"] = separate_qa
    akt_encoder_layer_config["dropout"] = dropout

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        train_strategy = local_params["train_strategy"]
        use_early_stop = local_params["use_early_stop"]
        epoch_early_stop = local_params["epoch_early_stop"]
        use_last_average = local_params["use_last_average"]
        epoch_last_average = local_params["epoch_last_average"]
        num_epoch = local_params["num_epoch"]
        # main_metric = local_params["main_metric"]
        # use_multi_metrics = local_params["use_multi_metrics"]
        # mutil_metrics = local_params["multi_metrics"]
        if train_strategy == "valid_test":
            if use_early_stop:
                pick_up_model_str = f"early_stop_{num_epoch}_{epoch_early_stop}"
            else:
                pick_up_model_str = f"num_epoch_{num_epoch}"
        elif train_strategy == "no_valid":
            if use_last_average:
                pick_up_model_str = f"last_average_{num_epoch}_{epoch_last_average}"
            else:
                pick_up_model_str = f"last_average_{num_epoch}"
        else:
            raise NotImplementedError()

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '-').replace(':', '-')}@@AKT@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}@@{train_strategy}@@{pick_up_model_str}"
            f"@@{num_concept}-{num_question}-{dim_model}-{key_query_same}-{num_block}-{num_head}-"
            f"{dim_ff}-{dim_final_fc}-{separate_qa}-{dropout}")


def akt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    save_params(global_params, global_objects)

    return global_params, global_objects


def akt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    duo_cl_general_config(local_params, global_params)
    save_params(global_params, global_objects)

    return global_params, global_objects


def akt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    instance_cl_general_config(local_params, global_params, global_objects)
    save_params(global_params, global_objects)

    return global_params, global_objects


def akt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    cluster_cl_general_config(local_params, global_params, global_objects)
    save_params(global_params, global_objects)

    return global_params, global_objects


def akt4cold_start_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)

    # 冷启动参数
    cold_start_step1 = local_params["cold_start_step1"]
    cold_start_step2 = local_params["cold_start_step2"]
    effect_start_step2 = local_params["effect_start_step2"]

    encoder_config_original = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    global_params["models_config"]["kt_model"]["encoder_layer"]["AKT4cold_start"] = deepcopy(encoder_config_original)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "AKT4cold_start"
    encoder_config_cold_start = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT4cold_start"]
    encoder_config_cold_start["cold_start_step1"] = cold_start_step1
    encoder_config_cold_start["cold_start_step2"] = cold_start_step2
    encoder_config_cold_start["effect_start_step2"] = effect_start_step2
    global_params["save_model_dir_name"] = global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT4cold_start@@")
    global_params["save_model_dir_name"] += f"@@{cold_start_step1}-{cold_start_step2}-{effect_start_step2}"
    save_params(global_params, global_objects)

    return global_params, global_objects
