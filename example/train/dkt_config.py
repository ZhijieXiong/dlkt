from copy import deepcopy
from config import general_config


from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS


def dkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["interaction"] = [num_concept * 2, dim_emb]

    # encoder layer
    dkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
    dkt_encoder_layer_config["dim_emb"] = dim_emb
    dkt_encoder_layer_config["dim_latent"] = dim_latent
    dkt_encoder_layer_config["rnn_type"] = rnn_type
    dkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = num_concept

    return global_params, global_objects