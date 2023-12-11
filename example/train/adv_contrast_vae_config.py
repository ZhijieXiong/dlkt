from copy import deepcopy

from _config import *

from lib.template.params_template_v2 import PARAMS
from lib.template.model.AC_VAE import MODEL_PARAMS as AC_VAE_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def adv_contrast_vae_gru_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(AC_VAE_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dim_rnn = local_params["dim_rnn"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    encoder_type = local_params["encoder_type"]
    dim_latent = local_params["dim_latent"]
    add_eps = local_params["add_eps"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["concept"] = [num_concept, dim_concept]
    kt_embed_layer_config["question"] = [num_question, dim_question]

    # rnn layer
    rnn_layer_config = global_params["models_config"]["kt_model"]["rnn_layer"]
    rnn_layer_config["dim_concept"] = dim_concept
    rnn_layer_config["dim_question"] = dim_question
    rnn_layer_config["dim_correct"] = dim_correct
    rnn_layer_config["dim_rnn"] = dim_rnn
    rnn_layer_config["rnn_type"] = rnn_type
    rnn_layer_config["num_rnn_layer"] = num_rnn_layer

    # encoder layer
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]
    encoder_layer_config["type"] = encoder_type
    encoder_layer_config["dim_latent"] = dim_latent
    encoder_layer_config["add_eps"] = add_eps

    # 配置dual和prior优化器参数
    config_optimizer(local_params, global_params, "dual")
    config_optimizer(local_params, global_params, "prior")

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    if data_type == "only_question":
        global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, "multi_concept")
    else:
        global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, "single_concept")

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '-').replace(':', '-')}@@AC_VAE_GRU@@seed_{local_params['seed']}@@"
            f"{setting_name}@@{train_file_name.replace('.txt', '')}@@{num_concept}-{num_question}-{dim_concept}-"
            f"{dim_question}-{dim_correct}-{dim_rnn}-{rnn_type}-{num_rnn_layer}-{encoder_type}-{dim_latent}-{add_eps}")


def adv_contrast_vae_gru_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    adv_contrast_vae_gru_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
