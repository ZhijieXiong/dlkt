from _config import *
from _cl_config import *
from _data_aug_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def qdkt_general_config(local_params, global_params):
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
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["concept"] = [num_concept, dim_concept]
    kt_embed_layer_config["question"] = [num_question, dim_question]

    # encoder layer
    qdkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    qdkt_encoder_layer_config["dim_concept"] = dim_concept
    qdkt_encoder_layer_config["dim_question"] = dim_question
    qdkt_encoder_layer_config["dim_correct"] = dim_correct
    qdkt_encoder_layer_config["dim_latent"] = dim_latent
    qdkt_encoder_layer_config["rnn_type"] = rnn_type
    qdkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_concept + dim_question
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '-').replace(':', '-')}@@qDKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}@@{num_concept}-{num_question}-{dim_concept}-{dim_question}-"
            f"{dim_correct}-{dim_latent}-{rnn_type}-{num_rnn_layer}-{dropout}-{num_predict_layer}-{dim_predict_mid}-"
            f"{activate_type}")


def qdkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params)
    params_str = instance_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-instance_cl@@"))
        global_params["save_model_dir_name"] += f"@@{params_str}"
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params)
    params_str = duo_cl_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-duo_cl@@"))
        global_params["save_model_dir_name"] += f"@@{params_str}"
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params)
    params_str = cluster_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-cluster_cl@@"))
        global_params["save_model_dir_name"] += f"@@{params_str}"
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params)
    params_str = max_entropy_adv_aug_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-ME_adv_aug@@"))
        global_params["save_model_dir_name"] += f"@@{params_str}"
        save_params(global_params, global_objects)

    return global_params, global_objects
