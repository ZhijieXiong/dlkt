from ._config import *

from lib.util.basic import *


def dkt_que_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "DKT_QUE",
                "DKT_QUE": {}
            },
            "predict_layer": {
                "type": "direct",
                "direct": {}
            }
        }
    }

    # 配置模型参数
    dataset_name = local_params["dataset_name"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    que_emb_file_name = local_params["que_emb_file_name"]
    frozen_que_emb = local_params["frozen_que_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    que_emb_path = os.path.join(
        global_objects["file_manager"].get_preprocessed_dir(dataset_name), que_emb_file_name
    )
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["question"] = {
        "num_item": num_question,
        "dim_item": dim_emb,
        "learnable": not frozen_que_emb,
        "embed_path": que_emb_path
    }

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKT_QUE"]
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_emb
    predict_layer_config["direct"]["dim_predict_out"] = 1

    global_objects["logger"].info(
        f"model params\n    "
        f"num_question: {num_question}, que_emb_path: {que_emb_path}, frozen_que_emb: {frozen_que_emb}\n    "
        f"dim_emb: {dim_emb}, dim_latent: {dim_latent}, rnn_type: {rnn_type}, num_rnn_layer: {num_rnn_layer}\n    "
        f"dropout: {dropout}, num_predict_layer: {num_predict_layer}, dim_predict_mid: {dim_predict_mid}, "
        f"activate_type: {activate_type}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DKT_QUE@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dkt_que_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dkt_que_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
