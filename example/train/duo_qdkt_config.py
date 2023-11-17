from copy import deepcopy
from config import general_config

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS


def duo_qdkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

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

    # 配置数据集参数
    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = "semantic_aug"
    datasets_train_config["kt4aug"]["num_aug"] = 1

    # 损失权重
    global_params["loss_config"]["cl loss"] = local_params["weight_cl_loss"]

    # 对比学习温度系数
    global_params["other"]["duo"]["temp"] = local_params["temp"]

    return global_params, global_objects
