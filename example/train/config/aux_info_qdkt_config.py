import os.path

from ._config import *
from ._cl_config import *

from lib.template.kt_model.AuxInfoQDKT import MODEL_PARAMS as AuxInfoQDKT_MODEL_PARAMS
from lib.util.basic import *


def aux_info_qdkt_general_config(local_params, global_params, global_objects):
    # 数据集特殊配置；主要是对use time和interval time进行聚合，区别于原始的LPKT，减少time的embedding数量
    global_params["datasets_config"]["train"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["train"]["agg_aux_info"] = {}
    global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["test"]["agg_aux_info"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "agg_aux_info"
        global_params["datasets_config"]["valid"]["agg_aux_info"] = {}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(AuxInfoQDKT_MODEL_PARAMS)

    # 配置模型参数
    dataset_name = local_params["dataset_name"]
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_question = local_params["dim_question"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]
    pretrain_aux_emb_path = local_params.get("pretrain_aux_emb_path", "")
    use_pretrain_aux_emb = os.path.exists(pretrain_aux_emb_path)

    # embed layer
    embed_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    embed_config["concept"] = [num_concept, dim_question]
    embed_config["question"] = [num_question, dim_question]
    embed_config["correct"] = [2, dim_question]
    embed_config["use_LLM_emb"] = False
    embed_config["use_pretrain_aux_emb"] = use_pretrain_aux_emb
    embed_config["pretrain_aux_emb_path"] = pretrain_aux_emb_path

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoQDKT"]
    encoder_config["dataset_name"] = dataset_name
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_question * 2 + dim_latent
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    # IPS
    use_sample_weight = local_params["use_sample_weight"]
    sample_weight_method = local_params["sample_weight_method"]
    IPS_min = local_params["IPS_min"]
    IPS_his_seq_len = local_params['IPS_his_seq_len']

    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["IPS_min"] = IPS_min
    global_params["IPS_his_seq_len"] = IPS_his_seq_len

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, num_correct: 2\n    "
        f"dim_question: {dim_question}, dim_latent: {dim_latent}, rnn_type: {rnn_type}, num_rnn_layer: {num_rnn_layer}, "
        f"dropout: {dropout},  num_predict_layer: {num_predict_layer}, dim_predict_mid: {dim_predict_mid}, "
        f"activate_type: {activate_type}\n    "
        f"use_pretrain_aux_emb: {use_pretrain_aux_emb}"
        f"{'' if not use_pretrain_aux_emb else f', pretrain_aux_emb_path: {pretrain_aux_emb_path}'}\n"
        f"IPS params\n    "
        f"use IPS: {use_sample_weight}, IPS_min: {IPS_min}, IPS_his_seq_len: {IPS_his_seq_len}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@AuxInfoQDKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def aux_info_qdkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    aux_info_qdkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
