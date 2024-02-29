from ._config import *

from lib.template.kt_model.ATKT import MODEL_PARAMS
from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS


def atkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "ATKT"

    # 配置模型参数
    use_concept = local_params["use_concept"]
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_correct = local_params["dim_correct"]
    dim_latent = local_params["dim_latent"]
    dim_attention = local_params["dim_attention"]
    dropout = local_params["dropout"]
    epsilon = local_params["epsilon"]
    beta = local_params["beta"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["ATKT"]
    encoder_config["use_concept"] = use_concept
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_concept"] = dim_concept
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dim_latent"] = dim_latent
    encoder_config["dim_attention"] = dim_attention
    encoder_config["dropout"] = dropout
    encoder_config["epsilon"] = epsilon

    global_params["loss_config"]["adv loss"] = beta

    global_objects["logger"].info(
        "model params\n"
        f"    use_concept: {use_concept}, num of concept: {num_concept}, num of question: {num_question}\n"
        f"    dim of concept emb: {dim_concept}, dim of correct emb: {dim_correct}, "
        f"dim of latent: {dim_latent}, dim of attention: {dim_attention}\n    dropout: {dropout}, epsilon: {epsilon}, beta: {beta}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@ATKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def atkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    atkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
