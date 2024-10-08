from ._config import *

from lib.template.params_template_v2 import PARAMS
from lib.template.kt_model.DKT import MODEL_PARAMS as DKT_MODEL_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def dkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(DKT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DKT"
    data_type = global_params["datasets_config"]["data_type"]

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]
    use_concept = local_params["use_concept"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    if use_concept and data_type == "only_question":
        # 将知识点当作习题（多个知识点构成一个习题）
        kt_embed_layer_config["concept"] = [num_concept, dim_emb]
    elif use_concept:
        kt_embed_layer_config["interaction"] = [num_concept * 2, dim_emb]
    else:
        kt_embed_layer_config["question"] = [num_question, dim_emb]

    # encoder layer
    dkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
    dkt_encoder_layer_config["num_concept"] = num_concept
    dkt_encoder_layer_config["num_question"] = num_question
    dkt_encoder_layer_config["dim_emb"] = dim_emb
    dkt_encoder_layer_config["dim_latent"] = dim_latent
    dkt_encoder_layer_config["rnn_type"] = rnn_type
    dkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer
    dkt_encoder_layer_config["use_concept"] = use_concept

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    # 当使用知识点输入，并且data type为multi 或者 single concept时，预测层输出为num concept，即每个知识点上的状态
    # 其它情况都是RNN输出的latent拼接上对应知识点（1、多个知识点embedding平均；2、对于像assist2015数据集，只有习题，则是习题emb）
    # 然后送入预测层，直接输出score
    if use_concept and data_type != "only_question":
        predict_layer_config["direct"]["dim_predict_in"] = dim_latent
        predict_layer_config["direct"]["dim_predict_out"] = num_concept
    else:
        predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_emb
        predict_layer_config["direct"]["dim_predict_out"] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    use concept: {use_concept}, num of concept: {num_concept}, num of question: {num_question}, \n"
        f"    dim of emb: {dim_emb}, dim of latent: {dim_latent}, rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}\n"
        f"    dropout: {dropout}, num of predict layer: {num_predict_layer}, dim of middle predict layer: {dim_predict_mid}, type of activate function: {activate_type}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
