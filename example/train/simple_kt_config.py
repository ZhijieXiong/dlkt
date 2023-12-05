from _config import *
from _cl_config import *
from _data_aug_config import *

from lib.template.params_template_v2 import PARAMS
from lib.template.model.SimpleKT import MODEL_PARAMS as SimpleKT_MODEL_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def simple_kt_general_config(local_params, global_params):
    global_params["models_config"]["kt_model"] = deepcopy(SimpleKT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "SimpleKT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    num_block = local_params["num_block"]
    num_head = local_params["num_head"]
    seq_len = local_params["seq_len"]
    dropout = local_params["dropout"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    dim_final_fc2 = local_params["dim_final_fc2"]
    key_query_same = local_params["key_query_same"]
    separate_qa = local_params["separate_qa"]
    difficulty_scalar = local_params["difficulty_scalar"]

    # encoder layer
    qdkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
    qdkt_encoder_layer_config["num_concept"] = num_concept
    qdkt_encoder_layer_config["num_question"] = num_question
    qdkt_encoder_layer_config["dim_model"] = dim_model
    qdkt_encoder_layer_config["num_block"] = num_block
    qdkt_encoder_layer_config["num_head"] = num_head
    qdkt_encoder_layer_config["dim_ff"] = dim_ff
    qdkt_encoder_layer_config["dim_final_fc"] = dim_final_fc
    qdkt_encoder_layer_config["dim_final_fc2"] = dim_final_fc2
    qdkt_encoder_layer_config["dropout"] = dropout
    qdkt_encoder_layer_config["seq_len"] = seq_len
    qdkt_encoder_layer_config["key_query_same"] = key_query_same
    qdkt_encoder_layer_config["separate_qa"] = separate_qa
    qdkt_encoder_layer_config["difficulty_scalar"] = difficulty_scalar

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '-').replace(':', '-')}@@SimpleKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}@@{num_concept}-{num_question}-{num_block}-{num_head}-{dim_ff}-"
            f"{dim_final_fc}-{dim_final_fc2}-{dropout}-{seq_len}-{key_query_same}-{separate_qa}-{difficulty_scalar}")


def simple_kt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    simple_kt_general_config(local_params, global_params)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def simple_kt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    simple_kt_general_config(local_params, global_params)
    params_str = cluster_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@SimpleKT@@", "@@SimpleKT-cluster_cl@@"))
        global_params["save_model_dir_name"] += f"@@{params_str}"
        save_params(global_params, global_objects)

    return global_params, global_objects
