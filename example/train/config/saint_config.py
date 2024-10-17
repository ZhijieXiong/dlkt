from ._config import *

from lib.util.basic import *


def saint_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "SAINT",
                "SAINT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    num_block = local_params["num_block"]
    num_attn_head = local_params["num_attn_head"]
    seq_len = local_params["seq_len"]
    dropout = local_params["dropout"]

    # backbone
    backbone_config = global_params["models_config"]["kt_model"]["encoder_layer"]["SAINT"]
    backbone_config["num_concept"] = num_concept
    backbone_config["num_question"] = num_question
    backbone_config["dim_emb"] = dim_emb
    backbone_config["num_block"] = num_block
    backbone_config["num_attn_head"] = num_attn_head
    backbone_config["seq_len"] = seq_len
    backbone_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, num_question: {num_question}, dim_emb: {dim_emb}, num_block: {num_block}, "
        f"num_attn_head: {num_attn_head}, dropout: {dropout}, seq_len: {seq_len}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"SAINT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def saint_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    saint_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
