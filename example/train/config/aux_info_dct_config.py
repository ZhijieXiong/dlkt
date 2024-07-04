from ._config import *
from ._cognition_tracing_config import *
from lib.template.kt_model.AuxInfoDCT import MODEL_PARAMS as AuxInfoDCT_PARAMS


def aux_info_dct_general_config(local_params, global_params, global_objects):
    # 数据集特殊配置；主要是对use time和interval time进行聚合，区别于原始的LPKT，减少time的embedding数量
    global_params["datasets_config"]["train"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["train"]["agg_aux_info"] = {}
    global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["test"]["agg_aux_info"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "agg_aux_info"
        global_params["datasets_config"]["valid"]["agg_aux_info"] = {}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(AuxInfoDCT_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    use_mean_pool4concept = local_params["use_mean_pool4concept"]
    use_proj = local_params["use_proj"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    num_mlp_layer = local_params["num_mlp_layer"]
    dropout = local_params["dropout"]
    max_que_disc = local_params["max_que_disc"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
    encoder_config["dataset_name"] = local_params["dataset_name"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["use_mean_pool4concept"] = use_mean_pool4concept
    encoder_config["use_proj"] = use_proj
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["num_mlp_layer"] = num_mlp_layer
    encoder_config["dropout"] = dropout
    encoder_config["max_que_disc"] = max_que_disc

    # IPS
    use_sample_weight = local_params["use_sample_weight"]
    sample_weight_method = local_params["sample_weight_method"]
    IPS_min = local_params["IPS_min"]
    IPS_his_seq_len = local_params['IPS_his_seq_len']

    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["IPS_min"] = IPS_min
    global_params["IPS_his_seq_len"] = IPS_his_seq_len

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}\n    "
        f"use_proj: {use_proj}, use_mean_pool4concept: {use_mean_pool4concept}, dim_emb: {dim_emb}, "
        f"dim_latent: {dim_latent}, rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}, "
        f"num_mlp_layer: {num_mlp_layer}, dropout: {dropout}, max_que_disc: {max_que_disc}\n"
        f"IPS params\n    "
        f"use IPS: {use_sample_weight}, sample_weight_method: {sample_weight_method}, IPS_min: {IPS_min}, "
        f"IPS_his_seq_len: {IPS_his_seq_len}"
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
