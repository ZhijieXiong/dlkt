from ._config import *

from lib.template.params_template_v2 import PARAMS
from lib.template.model.DKVMN import MODEL_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *


def dkvmn_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DKVMN"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_key = local_params["dim_key"]
    dim_value = local_params["dim_value"]
    dropout = local_params["dropout"]
    use_concept = local_params["use_concept"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKVMN"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_key"] = dim_key
    encoder_config["dim_value"] = dim_value
    encoder_config["use_concept"] = use_concept

    global_objects["logger"].info(
        "model params\n"
        f"    use concept: {use_concept}, num of concept: {num_concept}, num of question: {num_question}, "
        f"dim of key: {dim_key}, dim of value: {dim_value}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@DKVMN@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def dkvmn_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dkvmn_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
