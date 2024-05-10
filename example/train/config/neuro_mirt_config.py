from ._config import *
from ._cognition_tracing_config import *
from lib.template.kt_model.NeuroMIRT import MODEL_PARAMS as NeuroMIRT_PARAMS


def neuro_mirt_general_config(local_params, global_params, global_objects):
    global_params["datasets_config"]["train"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["train"]["agg_aux_info"] = {}
    global_params["datasets_config"]["test"]["type"] = "agg_aux_info"
    global_params["datasets_config"]["test"]["agg_aux_info"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "agg_aux_info"
        global_params["datasets_config"]["valid"]["agg_aux_info"] = {}

    global_params["models_config"] = {}
    global_params["models_config"]["kt_model"] = deepcopy(NeuroMIRT_PARAMS)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_concept_combination = local_params["num_concept_combination"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    dropout = local_params["dropout"]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["NeuroMIRT"]
    encoder_config["dataset_name"] = local_params["dataset_name"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_concept_combination"] = num_concept_combination
    encoder_config["num_question"] = num_question
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dropout"] = dropout

    # # 对比学习
    # w_cl_loss = local_params["w_cl_loss"]
    # temp = local_params["temp"]
    # correct_noise = local_params["correct_noise"]
    # if w_cl_loss != 0:
    #     global_params["loss_config"]["cl loss"] = w_cl_loss
    #     global_params["other"]["instance_cl"] = {
    #         "temp": temp,
    #         "correct_noise": correct_noise
    #     }

    global_objects["logger"].info(
        # f"cl params\n    "
        # f"temp: {temp}, correct_noise: {correct_noise}, w_cl_loss: {w_cl_loss}\n"
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, dim_emb: {dim_emb}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@AuxInfoDCT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def neuro_mirt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    neuro_mirt_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
