from ._config import *

from lib.util.basic import *


def lpkt_general_config(local_params, global_params, global_objects):
    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_e = local_params["dim_e"]
    dim_k = local_params["dim_k"]
    dim_correct = local_params["dim_correct"]
    dropout = local_params["dropout"]

    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "LPKT",
                "LPKT": {}
            }
        }
    }
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
    encoder_layer_config["num_concept"] = num_concept
    encoder_layer_config["num_question"] = num_question
    encoder_layer_config["dim_e"] = dim_e
    encoder_layer_config["dim_k"] = dim_k
    encoder_layer_config["dim_correct"] = dim_correct
    encoder_layer_config["dropout"] = dropout
    encoder_layer_config["ablation_set"] = local_params["ablation_set"]

    # q matrix
    global_objects["LPKT"] = {}
    global_objects["LPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + 0.03
    q_matrix = global_objects["LPKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of e: {dim_e}, dim of k: {dim_k}, "
        f"dim of correct emb: {dim_correct}, dropout: {dropout}\n"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"LPKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def lpkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
