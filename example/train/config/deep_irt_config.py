from ._config import *

from lib.util.basic import *


def deep_irt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "DeepIRT",
                "DeepIRT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    dim_emb = local_params["dim_emb"]
    size_memory = local_params["size_memory"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DeepIRT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["dim_emb"] = dim_emb
    encoder_config["size_memory"] = size_memory
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, dim_emb: {dim_emb}, size_memory: {size_memory}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DeepIRT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def deep_irt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    deep_irt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
