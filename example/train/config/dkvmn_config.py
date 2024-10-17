from ._config import *

from lib.util.basic import *


def dkvmn_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "DKVMN",
                "DKVMN": {}
            }
        }
    }

    # 配置模型参数
    use_concept = local_params["use_concept"]
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_key = local_params["dim_key"]
    dim_value = local_params["dim_value"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKVMN"]
    encoder_config["use_concept"] = use_concept
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_key"] = dim_key
    encoder_config["dim_value"] = dim_value
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n    "
        f"use_concept: {use_concept}, num_concept: {num_concept}, num_question: {num_question}, dim_key: {dim_key}, "
        f"dim_value: {dim_value}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DKVMN@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dkvmn_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dkvmn_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
