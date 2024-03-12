from ._config import *
from ._cognition_tracing_config import *
from ._data_aug_config import *
from lib.template.kt_model.DCT import MODEL_PARAMS as DCT_MODEL_PARAMS


def dct_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(DCT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DCT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    que_user_share_proj = local_params["que_user_share_proj"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer
    encoder_config["que_user_share_proj"] = que_user_share_proj
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
          f"model params\n"
          f"    num_concept: {num_concept}, num_question: {num_question}\n"
          f"    dim_question: {dim_question}, dim_correct: {dim_correct}, dim_latent: {dim_latent}, rnn type: {rnn_type}, "
          f"num of rnn layer: {num_rnn_layer}, que_user_share_proj: {que_user_share_proj}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@DCT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def dct_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dct_general_config(local_params, global_params, global_objects)
    cognition_tracing_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def dct_mex_entropy_aug_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dct_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@DCT@@", "@@DCT-ME-ADA@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
