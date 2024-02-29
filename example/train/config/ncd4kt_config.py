from ._config import *
from lib.template.params_template import PARAMS
from lib.template.kt_model.NCD4KT import MODEL_PARAMS as NCD4KT_MODEL_PARAMS
from lib.template.objects_template import OBJECTS


def ncd4kt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(NCD4KT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "NCD4KT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_correct = local_params["dim_correct"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # encoder layer
    qdkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["NCD4KT"]
    qdkt_encoder_layer_config["num_concept"] = num_concept
    qdkt_encoder_layer_config["num_question"] = num_question
    qdkt_encoder_layer_config["dim_concept"] = dim_concept
    qdkt_encoder_layer_config["dim_correct"] = dim_correct
    qdkt_encoder_layer_config["rnn_type"] = rnn_type
    qdkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = num_concept
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    # print("model params\n"
    #       f"    num of concept: {num_concept}, num of question: {num_question}, dim of question emb: {dim_question}, "
    #       f"dim of concept emb: {dim_concept}, dim of correct emb: {dim_correct}, dim of latent: {dim_latent}\n"
    #       f"    rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}, dropout: {dropout}, num of predict layer: {num_predict_layer}, "
    #       f"dim of middle predict layer: {dim_predict_mid}, type of activate function: {activate_type}\n")

    # if local_params["save_model"]:
    #     setting_name = local_params["setting_name"]
    #     train_file_name = local_params["train_file_name"]
    #
    #     global_params["save_model_dir_name"] = (
    #         f"{get_now_time().replace(' ', '-').replace(':', '-')}@@qDKT@@seed_{local_params['seed']}@@{setting_name}@@"
    #         f"{train_file_name.replace('.txt', '')}")


def ncd4kt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    ncd4kt_general_config(local_params, global_params, global_objects)
    # if local_params["save_model"]:
    #     save_params(global_params, global_objects)

    return global_params, global_objects
