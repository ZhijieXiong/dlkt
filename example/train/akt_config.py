from _config import *
from _cl_config import *
from _data_aug_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS as PARAMS2
from lib.template.model.AKT import MODEL_PARAMS as AKT_MODEL_PARAMS
from lib.util.basic import *


def akt_general_config(local_params, global_params):
    global_params["models_config"]["kt_model"] = deepcopy(AKT_MODEL_PARAMS)
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
    seq_representation = local_params["seq_representation"]

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
    akt_encoder_layer_config["seq_representation"] = seq_representation

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]

    print("model params\n"
          f"    num of concept: {num_concept}, num of question: {num_question}, dim of model: {dim_model}, num of block: {num_block}, "
          f"num of attention head: {num_head}, dim of ff: {dim_ff}, dim of final fc: {dim_final_fc}, dropout: {dropout}\n"
          f"    separate question answer: {separate_qa}, key and query of attention are same: {key_query_same}, representation of seq in AKT: {seq_representation}")

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@AKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def akt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    duo_cl_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT-duo_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    instance_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT-instance_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    cluster_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT-cluster_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    max_entropy_adv_aug_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT-ME_adv_aug@@"))
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

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT4cold_start@@")
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_meta_optimize_cl_config(local_params):
    global_params = deepcopy(PARAMS2)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params)
    meta_optimize_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@AKT@@", "@@AKT-meta_optimize_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
