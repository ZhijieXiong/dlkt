from ._config import *

from lib.util.basic import *


def sparse_kt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = {
        "encoder_layer": {
            "type": "SparseKT",
            "SparseKT": {}
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    difficulty_scalar = local_params["difficulty_scalar"]
    dim_model = local_params["dim_model"]
    num_block = local_params["num_block"]
    num_head = local_params["num_head"]
    dropout = local_params["dropout"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    dim_final_fc2 = local_params["dim_final_fc2"]
    kq_same = local_params["kq_same"]
    separate_qa = local_params["separate_qa"]
    seq_len = local_params["seq_len"]
    k_index = local_params["k_index"]
    use_akt_rasch = local_params["use_akt_rasch"]

    # encoder layer
    encode_config = global_params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
    encode_config["num_concept"] = num_concept
    encode_config["num_question"] = num_question
    encode_config["dim_model"] = dim_model
    encode_config["num_block"] = num_block
    encode_config["num_head"] = num_head
    encode_config["dim_ff"] = dim_ff
    encode_config["dim_final_fc"] = dim_final_fc
    encode_config["dim_final_fc2"] = dim_final_fc2
    encode_config["dropout"] = dropout
    encode_config["seq_len"] = seq_len
    encode_config["kq_same"] = kq_same
    encode_config["separate_qa"] = separate_qa
    encode_config["difficulty_scalar"] = difficulty_scalar
    encode_config["k_index"] = k_index
    encode_config["use_akt_rasch"] = use_akt_rasch

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_model: {dim_model}, num_block: {num_block}, "
        f"num_head: {num_head}, dim_ff: {dim_ff}, dim_final_fc: {dim_final_fc}, dim_final_fc2: {dim_final_fc2}\n    "
        f"dropout: {dropout}, seq_len: {seq_len}, separate_qa: {separate_qa}, kq_same: {kq_same}, "
        f"difficulty_scalar: {difficulty_scalar}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"SparseKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def sparse_kt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    sparse_kt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
