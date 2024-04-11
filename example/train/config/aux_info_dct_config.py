from ._config import *
from ._cognition_tracing_config import *
from lib.template.kt_model.AuxInfoDCT import MODEL_PARAMS as AuxInfoDCT_PARAMS


def aux_info_dct_general_config(local_params, global_params, global_objects):
    # 数据集特殊配置；主要是对use time和interval time进行聚合，区别于原始的LPKT，减少time的embedding数量
    global_params["datasets_config"]["train"]["type"] = "kt4aux_info"
    global_params["datasets_config"]["train"]["kt4aux_info"] = {}
    global_params["datasets_config"]["test"]["type"] = "kt4aux_info"
    global_params["datasets_config"]["test"]["kt4aux_info"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "kt4aux_info"
        global_params["datasets_config"]["valid"]["kt4aux_info"] = {}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(AuxInfoDCT_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_question = local_params["dim_question"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    num_mlp_layer = local_params["num_mlp_layer"]
    dropout = local_params["dropout"]
    weight_aux_emb = local_params["weight_aux_emb"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
    encoder_config["dataset_name"] = local_params["dataset_name"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_question"] = dim_question
    encoder_config["weight_aux_emb"] = weight_aux_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["num_mlp_layer"] = num_mlp_layer
    encoder_config["dropout"] = dropout

    # 对比学习
    w_cl_loss = local_params["w_cl_loss"]
    temp = local_params["temp"]
    if w_cl_loss != 0:
        global_params["loss_config"]["cl loss"] = w_cl_loss
        global_params["other"]["instance_cl"] = {
            "temp": temp
        }

    global_objects["logger"].info(
        f"cl params\n    "
        f"temp: {temp}, w_cl_loss: {w_cl_loss}\n"
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}\n    "
        f"weight_aux_emb: {weight_aux_emb}, dim_question: {dim_question}, dim_latent: {dim_latent}, "
        f"rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}, num_mlp_layer: {num_mlp_layer}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@AuxInfoDCT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def aux_info_dct_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    aux_info_dct_general_config(local_params, global_params, global_objects)
    cognition_tracing_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
