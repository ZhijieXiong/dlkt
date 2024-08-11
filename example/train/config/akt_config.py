from ._config import *
from ._cl_config import *
from ._data_aug_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS as PARAMS2
from lib.template.kt_model.AKT import MODEL_PARAMS as AKT_MODEL_PARAMS
from lib.util.basic import *


def akt_general_config(local_params, global_params, global_objects):
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

    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, num_question: {num_question}, dim_model: {dim_model}, num_block: {num_block}, "
        f"num_head: {num_head}, dim_ff: {dim_ff}, dim_final_fc: {dim_final_fc}, \n     dropout: {dropout}, "
        f"separate_qa: {separate_qa}, key_query_same: {key_query_same}, seq_representation: {seq_representation}\n"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"AKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def akt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)

    # IPS
    use_sample_weight = local_params["use_sample_weight"]
    sample_weight_method = local_params["sample_weight_method"]
    IPS_min = local_params["IPS_min"]
    IPS_his_seq_len = local_params['IPS_his_seq_len']

    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["IPS_min"] = IPS_min
    global_params["IPS_his_seq_len"] = IPS_his_seq_len

    global_objects["logger"].info(
        f"IPS params\n    "
        f"use IPS: {use_sample_weight}, IPS_min: {IPS_min}, IPS_his_seq_len: {IPS_his_seq_len}"
    )

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_core_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)

    # 配置模型参数
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "AKT",
                "AKT": {

                }
            }
        }
    }

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
    fusion_mode = local_params["fusion_mode"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_model"] = dim_model
    encoder_config["key_query_same"] = key_query_same
    encoder_config["num_head"] = num_head
    encoder_config["num_block"] = num_block
    encoder_config["dim_ff"] = dim_ff
    encoder_config["dim_final_fc"] = dim_final_fc
    encoder_config["separate_qa"] = separate_qa
    encoder_config["dropout"] = dropout
    encoder_config["seq_representation"] = seq_representation
    encoder_config["fusion_mode"] = fusion_mode

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]
    global_params["loss_config"]["KL loss"] = 1

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_model: {dim_model}, num_block: {num_block}, "
        f"num_head: {num_head}, dim_ff: {dim_ff}, dim_final_fc: {dim_final_fc}, \n     dropout: {dropout}, "
        f"separate_qa: {separate_qa}, key_query_same: {key_query_same}, seq_representation: {seq_representation}\n    "
        f"fusion_mode: {fusion_mode}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"AKT-CORE@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    duo_cl_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-DUO-CL@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    instance_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-instance-CL@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    cluster_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-cluster-CL@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-ME-ADA@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt4cold_start_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)

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
        global_params["save_model_dir_name"] = global_params["save_model_dir_name"].replace("AKT@@", "AKT-cold-start@@")
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_meta_optimize_cl_config(local_params):
    global_params = deepcopy(PARAMS2)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    meta_optimize_cl_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-meta-optimize-CL@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_output_enhance_config(local_params):
    global_params = deepcopy(PARAMS2)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    output_enhance_general_config(local_params, global_params, global_objects)
    global_params["datasets_config"]["train"]["kt_output_enhance"] = {}
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("AKT@@", "AKT-output-enhance@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_LfF_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)

    q = local_params["q"]
    global_params["other"]["LfF"] = {
        "q": q
    }

    # 配置两个优化器的参数（使用相同的参数）
    config_optimizer(local_params, global_params, global_objects, "model_biased", same_as_kt=True)
    config_optimizer(local_params, global_params, global_objects, "model_de_biased", same_as_kt=True)

    global_objects["logger"].info(
        f"LfF params\n    "
        f"q: {q}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"AKT-LfF@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )
        save_params(global_params, global_objects)

    return global_params, global_objects


def akt_adv_bias_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    akt_general_config(local_params, global_params, global_objects)
    adv_bias_aug_general_config(local_params, global_params, global_objects)

    # IPS
    use_sample_weight = local_params["use_sample_weight"]
    sample_weight_method = local_params["sample_weight_method"]
    IPS_min = local_params["IPS_min"]
    IPS_his_seq_len = local_params['IPS_his_seq_len']

    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["IPS_min"] = IPS_min
    global_params["IPS_his_seq_len"] = IPS_his_seq_len

    global_objects["logger"].info(
        f"IPS params\n    "
        f"use IPS: {use_sample_weight}, IPS_min: {IPS_min}, IPS_his_seq_len: {IPS_his_seq_len}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"AKT-ADA@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}"
        )
        save_params(global_params, global_objects)

    return global_params, global_objects
