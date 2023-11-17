from copy import deepcopy
from config import general_config


from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS


def akt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "AKT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    key_query_same = local_params["key_query_same"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    separate_qa = local_params["separate_qa"]
    dropout = local_params["dropout"]

    # encoder layer
    akt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    akt_encoder_layer_config["num_concept"] = num_concept
    akt_encoder_layer_config["num_question"] = num_question
    akt_encoder_layer_config["dim_model"] = dim_model
    akt_encoder_layer_config["key_query_same"] = key_query_same
    akt_encoder_layer_config["num_head"] = num_head
    akt_encoder_layer_config["num_block"] = num_block
    akt_encoder_layer_config["dim_ff"] = dim_ff
    akt_encoder_layer_config["dim_final_fc"] = dim_final_fc
    akt_encoder_layer_config["separate_qa"] = separate_qa
    akt_encoder_layer_config["dropout"] = dropout

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]

    return global_params, global_objects
