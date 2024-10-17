from ._config import *

from lib.util.basic import *


def sakt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "SAKT",
                "SAKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    dim_emb = local_params["dim_emb"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    seq_len = local_params["seq_len"]
    dropout = local_params["dropout"]

    encode_config = global_params["models_config"]["kt_model"]["encoder_layer"]["SAKT"]
    encode_config["num_concept"] = num_concept
    encode_config["dim_emb"] = dim_emb
    encode_config["num_head"] = num_head
    encode_config["num_block"] = num_block
    encode_config["seq_len"] = seq_len
    encode_config["dropout"] = dropout

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, dim_emb: {dim_emb}, num_block: {num_block}, num_head: {num_head}, "
        f"dropout: {dropout}, seq_len: {seq_len}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"SAKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def sakt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    sakt_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
