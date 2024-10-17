from ._config import *

from lib.util.basic import *


def ncd_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "cd_model": {
            "backbone": {
                "type": "NCD",
                "NCD": {}
            },
            "predict_layer": {}
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_user = local_params["num_user"]
    dim_predict1 = local_params["dim_predict1"]
    dim_predict2 = local_params["dim_predict2"]
    dropout = local_params["dropout"]

    # backbone
    backbone_config = global_params["models_config"]["cd_model"]["backbone"]["NCD"]
    backbone_config["num_concept"] = num_concept
    backbone_config["num_question"] = num_question
    backbone_config["num_user"] = num_user

    # predict layer
    predict_layer_config = global_params["models_config"]["cd_model"]["predict_layer"]
    predict_layer_config["dim_predict1"] = dim_predict1
    predict_layer_config["dim_predict2"] = dim_predict2
    predict_layer_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n"
        f"    num of user: {num_user}, num of concept: {num_concept}, num of question: {num_question}, "
        f"dim of predict layer 1: {dim_predict1}, dim of predict layer 2: {dim_predict2}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@NCD@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def ncd_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    ncd_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
