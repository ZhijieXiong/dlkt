from ._config import *
from lib.template.kt_model.ELMKT import MODEL_PARAMS as ELMKT_PARAMS


def elmkt_general_config(local_params, global_params, global_objects):
    global_params["datasets_config"]["train"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["train"]["agg_aux_info"] = {}
    global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["test"]["agg_aux_info"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "agg_aux_info"
        global_params["datasets_config"]["valid"]["agg_aux_info"] = {}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(ELMKT_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    use_learnable_q = local_params["use_learnable_q"]
    max_q_unrelated_concept = local_params["max_q_unrelated_concept"]
    min_q_related_concept = local_params["min_q_related_concept"]
    use_lpkt_predictor = local_params["use_lpkt_predictor"]
    num_predictor_layer = local_params["num_predictor_layer"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["ELMKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["use_learnable_q"] = use_learnable_q
    encoder_config["max_q_unrelated_concept"] = max_q_unrelated_concept
    encoder_config["min_q_related_concept"] = min_q_related_concept
    encoder_config["use_lpkt_predictor"] = use_lpkt_predictor
    encoder_config["num_predictor_layer"] = num_predictor_layer
    encoder_config["dropout"] = dropout

    # 对比学习
    w_cl_loss = local_params["w_cl_loss"]
    temp = local_params["temp"]
    correct_noise = local_params["correct_noise"]
    global_params["other"]["instance_cl"] = {
        "temp": temp,
        "correct_noise": correct_noise
    }
    if w_cl_loss != 0:
        global_params["loss_config"]["cl loss"] = w_cl_loss

    global_objects["logger"].info(
        f"cl params\n    "
        f"temp: {temp}, correct_noise: {correct_noise}, w_cl_loss: {w_cl_loss}\n"
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_emb: {dim_emb}, dropout: {dropout}\n    "
        f"use_learnable_q: {use_learnable_q}, max_q_unrelated_concept: {max_q_unrelated_concept}, "
        f"min_q_related_concept: {min_q_related_concept}, use_lpkt_predictor: {use_lpkt_predictor}, "
        f"num_predictor_layer: {num_predictor_layer}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@ELMKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def elmkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    elmkt_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
