from ._config import *

from lib.util.basic import *


def mikt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "MIKT",
                "MIKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    dim_state = local_params["dim_state"]
    dropout = local_params["dropout"]
    seq_len = local_params["seq_len"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["MIKT"]
    encoder_config["seq_len"] = seq_len
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_state"] = dim_state
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_emb: {dim_emb}, dim_state: {dim_state}, "
        f"dropout: {dropout}, seq_len: {seq_len}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"MIKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def mikt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    mikt_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
