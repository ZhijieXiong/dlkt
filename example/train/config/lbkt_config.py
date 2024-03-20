from ._config import *
from ._cognition_tracing_config import *

from lib.template.kt_model.LBKT import MODEL_PARAMS as LBKT_MODEL_PARAMS
from lib.util.basic import *


def lbkt_general_config(local_params, global_params, global_objects):
    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_correct = local_params["num_correct"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dropout = local_params["dropout"]
    dim_h = local_params["dim_h"]
    dim_factor = local_params["dim_factor"]
    r = local_params["r"]
    d = local_params["d"]
    k = local_params["k"]
    b = local_params["b"]
    q_gamma = local_params["q_gamma"]

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(LBKT_MODEL_PARAMS)
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LBKT"]
    encoder_layer_config["num_concept"] = num_concept
    encoder_layer_config["num_question"] = num_question
    encoder_layer_config["num_correct"] = num_correct
    encoder_layer_config["dim_question"] = dim_question
    encoder_layer_config["dim_correct"] = dim_correct
    encoder_layer_config["dropout"] = dropout
    encoder_layer_config["dim_h"] = dim_h
    encoder_layer_config["dim_factor"] = dim_factor
    encoder_layer_config["r"] = r
    encoder_layer_config["d"] = d
    encoder_layer_config["k"] = k
    encoder_layer_config["b"] = b

    # q matrix
    global_objects["LBKT"] = {}
    global_objects["LBKT"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + q_gamma
    q_matrix = global_objects["LBKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, num_correct: {num_correct}, \n    "
        f"dim_question: {dim_question}, dim_correct: {dim_correct}, dim_h: {dim_h}, dim_factor: {dim_factor}, "
        f"dropout: {dropout}, r: {r}, d: {d}, k:{k}, b:{b}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@LBKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def lbkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lbkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
