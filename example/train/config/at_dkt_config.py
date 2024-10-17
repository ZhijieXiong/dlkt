from ._config import *

from lib.util.basic import *


def at_dkt_general_config(local_params, global_params):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "AT_DKT",
                "AT_DKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    QT_net_type = local_params["QT_net_type"]
    QT_rnn_type = local_params["QT_rnn_type"]
    QT_num_rnn_layer = local_params["QT_num_rnn_layer"]
    QT_transformer_num_block = local_params["QT_transformer_num_block"]
    QT_transformer_num_head = local_params["QT_transformer_num_head"]
    IK_start = local_params["IK_start"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["concept"] = [num_concept, dim_emb]
    kt_embed_layer_config["question"] = [num_question, dim_emb]
    kt_embed_layer_config["interaction"] = [num_concept * 2, dim_emb]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["dropout"] = dropout
    encoder_config["QT_net_type"] = QT_net_type
    encoder_config["QT_rnn_type"] = QT_rnn_type
    encoder_config["QT_num_rnn_layer"] = QT_num_rnn_layer
    encoder_config["QT_transformer_num_block"] = QT_transformer_num_block
    encoder_config["QT_transformer_num_head"] = QT_transformer_num_head
    encoder_config["IK_start"] = IK_start

    # 辅助损失权重
    weight_QT_loss = local_params["weight_QT_loss"]
    weight_IK_loss = local_params["weight_IK_loss"]
    global_params["loss_config"]["QT loss"] = weight_QT_loss
    global_params["loss_config"]["IK loss"] = weight_IK_loss

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"AT-DKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def at_dkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    at_dkt_general_config(local_params, global_params)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
