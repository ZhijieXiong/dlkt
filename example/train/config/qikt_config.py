from ._config import *

from lib.util.basic import *


def qikt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "QIKT",
                "QIKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_mlp_layer = local_params["num_mlp_layer"]
    lambda_q_all = local_params["lambda_q_all"]
    lambda_c_next = local_params["lambda_c_next"]
    lambda_c_all = local_params["lambda_c_all"]
    use_irt = local_params["use_irt"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["concept"] = [num_concept, dim_emb]
    kt_embed_layer_config["question"] = [num_question, dim_emb]

    # encoder layer
    qikt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["QIKT"]
    qikt_encoder_layer_config["num_concept"] = num_concept
    qikt_encoder_layer_config["num_question"] = num_question
    qikt_encoder_layer_config["dim_emb"] = dim_emb
    qikt_encoder_layer_config["rnn_type"] = rnn_type
    qikt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer
    qikt_encoder_layer_config["dropout"] = dropout
    qikt_encoder_layer_config["num_mlp_layer"] = num_mlp_layer
    qikt_encoder_layer_config["lambda_q_all"] = lambda_q_all
    qikt_encoder_layer_config["lambda_c_next"] = lambda_c_next
    qikt_encoder_layer_config["lambda_c_all"] = lambda_c_all
    qikt_encoder_layer_config["use_irt"] = use_irt

    # 损失权重
    global_params["loss_config"]["q all loss"] = local_params["weight_predict_q_all_loss"]
    global_params["loss_config"]["q next loss"] = local_params["weight_predict_q_next_loss"]
    global_params["loss_config"]["c all loss"] = local_params["weight_predict_c_all_loss"]
    global_params["loss_config"]["c next loss"] = local_params["weight_predict_c_next_loss"]

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of emb: {dim_emb}, rnn type: {rnn_type}, "
        f"num of rnn layer: {num_rnn_layer}, num of mlp layer: {num_mlp_layer}, dropout: {dropout}\n"
        f"    lambda of q_all: {lambda_q_all}, lambda of c_next: {lambda_c_next}, lambda of c_all: {lambda_c_all}, use irt: {use_irt}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"QIKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def qikt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    qikt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
