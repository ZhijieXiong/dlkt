from ._config import *
from ._cognition_tracing_config import *
from lib.template.kt_model.LBMKT import MODEL_PARAMS as LBMKT_PARAMS


def lbmkt_general_config(local_params, global_params, global_objects):
    # 数据集特殊配置；主要是对use time和interval time进行聚合，区别于原始的LPKT，减少time的embedding数量
    global_params["datasets_config"]["train"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["train"]["agg_aux_info"] = {"agg_num": True}
    global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["test"]["agg_aux_info"] = {"agg_num": True}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "agg_aux_info"
        global_params["datasets_config"]["valid"]["agg_aux_info"] = {"agg_num": True}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(LBMKT_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    num_proj_layer = local_params["num_proj_layer"]
    dropout = local_params["dropout"]
    max_que_disc = local_params["max_que_disc"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LBMKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["num_proj_layer"] = num_proj_layer
    encoder_config["dropout"] = dropout
    encoder_config["max_que_disc"] = max_que_disc

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}\n    "
        f"dim_emb: {dim_emb}, num_proj_layer: {num_proj_layer}, dropout: {dropout}, max_que_disc: {max_que_disc}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@AuxInfoDCT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def lbmkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lbmkt_general_config(local_params, global_params, global_objects)
    cognition_tracing_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
