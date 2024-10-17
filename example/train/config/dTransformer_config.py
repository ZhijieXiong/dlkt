from ._config import *

from lib.util.basic import *


def dTransformer_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "DTransformer",
                "DTransformer": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    dim_final_fc = local_params["dim_final_fc"]
    num_knowledge_prototype = local_params["num_knowledge_prototype"]
    dropout = local_params["dropout"]
    num_layer = local_params["num_layer"]
    num_head = local_params["num_head"]
    window = local_params["window"]
    proj = local_params["proj"]
    use_question = local_params["use_question"]
    key_query_same = local_params["key_query_same"]
    bias = local_params["bias"]
    use_hard_neg = local_params["use_hard_neg"]
    temp = local_params["temp"]

    # embed layer
    global_params["models_config"] = {
        "kt_model": {
            "type": "DTransformer",
            "encoder_layer": {
                "DTransformer": {}
            }
        }
    }
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
    encoder_layer_config["num_concept"] = num_concept
    encoder_layer_config["num_question"] = num_question
    encoder_layer_config["dim_model"] = dim_model
    encoder_layer_config["dim_final_fc"] = dim_final_fc
    encoder_layer_config["num_knowledge_prototype"] = num_knowledge_prototype
    encoder_layer_config["num_layer"] = num_layer
    encoder_layer_config["num_head"] = num_head
    encoder_layer_config["dropout"] = dropout
    encoder_layer_config["window"] = window
    encoder_layer_config["proj"] = proj
    encoder_layer_config["use_question"] = use_question
    encoder_layer_config["key_query_same"] = key_query_same
    encoder_layer_config["bias"] = bias
    encoder_layer_config["use_hard_neg"] = use_hard_neg
    encoder_layer_config["temp"] = temp

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    weight_reg_loss = local_params["weight_reg_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss
    global_params["loss_config"]["reg loss"] = weight_reg_loss

    global_objects["logger"].info(
        "model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, temp: {temp}\n    "
        f"dim_model: {dim_model}, dim_final_fc: {dim_final_fc}, num_knowledge_prototype: {num_knowledge_prototype}, "
        f"num_layer: {num_layer}, num_head: {num_head}, dropout: {dropout}, window: {window}\n    "
        f"use_question: {use_question}, key_query_same: {key_query_same}, bias: {bias}, use_hard_neg: {use_hard_neg}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DTransformer@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dTransformer_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dTransformer_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
