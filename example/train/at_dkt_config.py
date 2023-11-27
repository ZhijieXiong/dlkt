from copy import deepcopy
from _config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def at_dkt_general_config(local_params, global_params):
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
    at_dkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
    at_dkt_encoder_layer_config["num_concept"] = num_concept
    at_dkt_encoder_layer_config["num_question"] = num_question
    at_dkt_encoder_layer_config["dim_emb"] = dim_emb
    at_dkt_encoder_layer_config["dim_latent"] = dim_latent
    at_dkt_encoder_layer_config["rnn_type"] = rnn_type
    at_dkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer
    at_dkt_encoder_layer_config["dropout"] = dropout
    at_dkt_encoder_layer_config["QT_net_type"] = QT_net_type
    at_dkt_encoder_layer_config["QT_rnn_type"] = QT_rnn_type
    at_dkt_encoder_layer_config["QT_num_rnn_layer"] = QT_num_rnn_layer
    at_dkt_encoder_layer_config["QT_transformer_num_block"] = QT_transformer_num_block
    at_dkt_encoder_layer_config["QT_transformer_num_head"] = QT_transformer_num_head
    at_dkt_encoder_layer_config["IK_start"] = IK_start

    # 辅助损失权重
    weight_QT_loss = local_params["weight_QT_loss"]
    weight_IK_loss = local_params["weight_IK_loss"]
    global_params["loss_config"]["QT_loss"] = weight_QT_loss
    global_params["loss_config"]["IK_loss"] = weight_IK_loss

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        train_strategy = local_params["train_strategy"]
        use_early_stop = local_params["use_early_stop"]
        epoch_early_stop = local_params["epoch_early_stop"]
        use_last_average = local_params["use_last_average"]
        epoch_last_average = local_params["epoch_last_average"]
        num_epoch = local_params["num_epoch"]

        if train_strategy == "valid_test":
            if use_early_stop:
                pick_up_model_str = f"early_stop_{num_epoch}_{epoch_early_stop}"
            else:
                pick_up_model_str = f"num_epoch_{num_epoch}"
        elif train_strategy == "no_valid":
            if use_last_average:
                pick_up_model_str = f"last_average_{num_epoch}_{epoch_last_average}"
            else:
                pick_up_model_str = f"last_average_{num_epoch}"
        else:
            raise NotImplementedError()

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '-').replace(':', '-')}@@AT-DKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}@@{train_strategy}@@{pick_up_model_str}"
            f"@@{num_concept}-{num_question}-{dim_emb}-{dim_latent}-{rnn_type}-{num_rnn_layer}-{dropout}-")
        QT_params_str = f"{QT_rnn_type}-{QT_num_rnn_layer}" if QT_net_type == 'rnn' else \
            f"transformer-{QT_transformer_num_block}-{QT_transformer_num_head}"
        global_params["save_model_dir_name"] += QT_params_str + f"-{IK_start}-{weight_QT_loss}-{weight_IK_loss}"


def at_dkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    at_dkt_general_config(local_params, global_params)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
