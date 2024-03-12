from ._config import *
from ._cognition_tracing_config import *

from lib.util.basic import *
from lib.CONSTANT import INTERVAL_TIME4LPKT_PLUS, USE_TIME4LPKT_PLUS


def lpkt_plus_general_config(local_params, global_params, global_objects):
    # 数据集特殊配置
    global_params["other"] = {"lpkt_plus": {}}
    global_params["datasets_config"]["train"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["train"]["lpkt_plus"] = {}
    global_params["datasets_config"]["test"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["test"]["kt4lpkt_plus"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "kt4lpkt_plus"
        global_params["datasets_config"]["valid"]["kt4lpkt_plus"] = {}

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_question = local_params["dim_question"]
    dim_latent = local_params["dim_latent"]
    dim_correct = local_params["dim_correct"]
    que_user_share_proj = local_params["que_user_share_proj"]
    dropout = local_params["dropout"]
    ablation_set = local_params["ablation_set"]

    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "LPKT+",
                "LPKT+": {}
            }
        }
    }
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LPKT+"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_latent"] = dim_latent
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dropout"] = dropout
    encoder_config["que_user_share_proj"] = que_user_share_proj
    encoder_config["ablation_set"] = ablation_set
    encoder_config["num_interval_time"] = len(INTERVAL_TIME4LPKT_PLUS)
    encoder_config["num_use_time"] = len(USE_TIME4LPKT_PLUS)

    global_objects["lpkt_plus"] = {}
    global_objects["logger"].info(
        "model params\n"
        f"    num_concept: {num_concept}, num_question: {num_question}, num_interval_time: {len(INTERVAL_TIME4LPKT_PLUS)}, "
        f"num_use_time: {len(USE_TIME4LPKT_PLUS)}, ablation_set: {ablation_set}\n"
        f"    dim_question: {dim_question}, dim_latent: {dim_latent}, dim_correct: {dim_correct}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@LPKT+@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def lpkt_plus_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lpkt_plus_general_config(local_params, global_params, global_objects)
    cognition_tracing_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
