import os
from copy import deepcopy
from _config import general_config
from _cl_config import *
from _data_aug_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *
from lib.util.data import write_json


def qdkt_general_config(local_params, global_params, global_objects):
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
        file_manager = global_objects["file_manager"]
        model_root_dir = file_manager.get_models_dir()
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        train_strategy = local_params["train_strategy"]
        use_early_stop = local_params["use_early_stop"]
        epoch_early_stop = local_params["epoch_early_stop"]
        use_last_average = local_params["use_last_average"]
        epoch_last_average = local_params["epoch_last_average"]
        num_epoch = local_params["num_epoch"]
        # main_metric = local_params["main_metric"]
        # use_multi_metrics = local_params["use_multi_metrics"]
        # mutil_metrics = local_params["multi_metrics"]
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

        model_dir_name = (f"{get_now_time().replace(' ', '-').replace(':', '-')}@@{setting_name}@@"
                          f"{train_file_name.replace('.txt', '')}@@{train_strategy}@@{pick_up_model_str}"
                          f"@@{num_concept}-{num_question}-{dim_concept}-{dim_question}-{dim_correct}-{dim_latent}-"
                          f"{rnn_type}-{num_rnn_layer}-{dropout}-{num_predict_layer}-{dim_predict_mid}-{activate_type}")
        model_dir = os.path.join(model_root_dir, model_dir_name)
        global_params["save_model_dir"] = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        else:
            assert False, f"{model_dir} exists"


def save_params(global_params):
    if global_params["save_model"]:
        params_path = os.path.join(global_params["save_model_dir"], "params.json")
        params_json = params2str(global_params)
        write_json(params_json, params_path)


def qdkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    save_params(global_params)

    return global_params, global_objects


def qdkt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    instance_cl_general_config(local_params, global_params, global_objects)
    save_params(global_params)

    return global_params, global_objects


def qdkt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    duo_cl_general_config(local_params, global_params)
    save_params(global_params)

    return global_params, global_objects


def qdkt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    cluster_cl_general_config(local_params, global_params, global_objects)
    save_params(global_params)

    return global_params, global_objects


def qdkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params)
    save_params(global_params, global_objects)

    return global_params, global_objects
