import os.path

from ._config import *

from lib.util.basic import *


def qdkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "qDKT",
                "qDKT": {}
            },
            "predict_layer": {
                "type": "direct",
                "direct": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    embed_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    embed_config["concept"] = [num_concept, dim_concept]
    embed_config["question"] = [num_question, dim_question]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    encoder_config["dim_concept"] = dim_concept
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_concept + dim_question
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of question emb: {dim_question}, "
        f"dim of concept emb: {dim_concept}, dim of correct emb: {dim_correct}, dim of latent: {dim_latent}\n"
        f"    rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}, dropout: {dropout}, num of predict layer: {num_predict_layer}, "
        f"dim of middle predict layer: {dim_predict_mid}, type of activate function: {activate_type}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"qDKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def qdkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)

    # sample_reweight
    sample_reweight = {
        "use_sample_reweight": local_params["use_sample_reweight"],
        "sample_reweight_method": local_params["sample_reweight_method"],
    }
    use_sample_reweight = local_params["use_sample_reweight"]
    sample_reweight_method = local_params["sample_reweight_method"]
    use_IPS = False
    if use_sample_reweight and local_params["sample_reweight_method"] in ["IPS-double", "IPS-seq", "IPS-question"]:
        use_IPS = True
        sample_reweight["IPS_min"] = local_params["IPS_min"]
        sample_reweight["IPS_his_seq_len"] = local_params['IPS_his_seq_len']
    global_params["sample_reweight"] = sample_reweight

    global_objects["logger"].info(
        f"sample weight\n    "
        f"use_sample_reweight: {use_sample_reweight}, sample_reweight_method: {sample_reweight_method}"
        f"{', IPS_min: ' + str(local_params['IPS_min']) + ', IPS_his_seq_len: ' + str(local_params['IPS_his_seq_len']) if use_IPS else ''}"
    )

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_core_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {

            },
            "encoder_layer": {
                "type": "qDKT_CORE",
                "qDKT_CORE": {

                }
            },
            "predict_layer": {
                "type": "direct",
                "direct": {

                }
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]
    fusion_mode = local_params["fusion_mode"]

    # embed layer
    embed_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    embed_config["concept"] = [num_concept, dim_concept]
    embed_config["question"] = [num_question, dim_question]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]
    encoder_config["dim_concept"] = dim_concept
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["fusion_mode"] = fusion_mode

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_concept + dim_question
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 2

    global_params["loss_config"]["KL loss"] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, num_question: {num_question}, dim_question: {dim_question}, "
        f"dim_concept: {dim_concept}, dim_correct: {dim_correct}, dim_latent: {dim_latent}\n"
        f"    rnn_type: {rnn_type}, num_rnn_layer: {num_rnn_layer}, dropout: {dropout}, num_predict_layer: {num_predict_layer}, "
        f"dim_predict_mid: {dim_predict_mid}, activate_type: {activate_type}, fusion_mode: {fusion_mode}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"qDKT-CORE-NEW@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")
        save_params(global_params, global_objects)

    return global_params, global_objects
