from ._config import *

from lib.util.basic import *


def simple_kt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "SimpleKT",
                "SimpleKT": {}
            }
        }
    }

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

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of model: {dim_model}, num of block: {num_block}, "
        f"num of attention head: {num_head}, dim of ff: {dim_ff}, dim of final fc: {dim_final_fc}, dim of final fc2: {dim_final_fc2} "
        f"dropout: {dropout}, length of sequence: {seq_len}\n"
        f"    separate question answer: {separate_qa}, key and query of attention are same: {key_query_same}, use scalar difficulty: {difficulty_scalar}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"SimpleKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def simple_kt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    simple_kt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
