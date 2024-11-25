from ._config import *

from lib.util.basic import *


def dkvmn_que_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "DKVMN_QUE",
                "DKVMN_QUE": {}
            }
        }
    }

    # 配置模型参数
    dataset_name = local_params["dataset_name"]
    num_question = local_params["num_question"]
    que_emb_file_name = local_params["que_emb_file_name"]
    frozen_que_emb = local_params["frozen_que_emb"]
    dim_key = local_params["dim_key"]
    dim_value = local_params["dim_value"]
    dropout = local_params["dropout"]

    # embed layer
    que_emb_path = os.path.join(
        global_objects["file_manager"].get_preprocessed_dir(dataset_name), que_emb_file_name
    )
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["question"] = {
        "num_item": num_question,
        "dim_item": dim_key,
        "learnable": not frozen_que_emb,
        "embed_path": que_emb_path
    }

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKVMN_QUE"]
    encoder_config["num_question"] = num_question
    encoder_config["dim_key"] = dim_key
    encoder_config["dim_value"] = dim_value
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n    "
        f"num_question: {num_question}, que_emb_path: {que_emb_path}, frozen_que_emb: {frozen_que_emb}\n    "
        f"dim_key: {dim_key}, dim_value: {dim_value}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DKVMN_QUE@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dkvmn_que_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dkvmn_que_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
