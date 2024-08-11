from ._config import *
from ._cognition_tracing_config import *
from lib.template.kt_model.DCT import MODEL_PARAMS as DCT_MODEL_PARAMS


def dct_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(DCT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DCT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    use_mean_pool4concept = local_params["use_mean_pool4concept"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    num_mlp_layer = local_params["num_mlp_layer"]
    dropout = local_params["dropout"]
    max_que_disc = local_params["max_que_disc"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["use_mean_pool4concept"] = use_mean_pool4concept
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["num_mlp_layer"] = num_mlp_layer
    encoder_config["dropout"] = dropout
    encoder_config["max_que_disc"] = max_que_disc

    global_objects["logger"].info(
          f"model params\n    "
          f"num_concept: {num_concept}, num_question: {num_question}\n    "
          f"use_mean_pool4concept: {use_mean_pool4concept}, dim_emb: {dim_emb}, dim_latent: {dim_latent}, "
          f"rnn_type: {rnn_type}, num_rnn_layer: {num_rnn_layer}, num_mlp_layer: {num_mlp_layer}, "
          f"max_que_disc: {max_que_disc}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DCT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dct_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dct_general_config(local_params, global_params, global_objects)
    cognition_tracing_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
