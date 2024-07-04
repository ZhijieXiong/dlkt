from ._config import *
from ._cl_config import *
from ._data_aug_config import *
from ._melt_config import *
from ._dro_config import *

from lib.template.params_template import PARAMS
from lib.template.params_template_v2 import PARAMS as PARAMS2
from lib.template.kt_model.qDKT import MODEL_PARAMS as qDKT_MODEL_PARAMS
from lib.template.kt_model.qDKT_CORE import MODEL_PARAMS as qDKT_CORE_MODEL_PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.basic import *
from lib.util.statics import cal_propensity


def qdkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(qDKT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "qDKT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_concept = local_params["dim_concept"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    embed_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    embed_config["concept"] = [num_concept, dim_concept]
    embed_config["question"] = [num_question, dim_question]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    encoder_config["dim_concept"] = dim_concept
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dim_latent"] = dim_latent
    encoder_config["rnn_type"] = rnn_type
    encoder_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_concept + dim_question
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of question emb: {dim_question}, "
        f"dim of concept emb: {dim_concept}, dim of correct emb: {dim_correct}, dim of latent: {dim_latent}\n"
        f"    rnn type: {rnn_type}, num of rnn layer: {num_rnn_layer}, dropout: {dropout}, num of predict layer: {num_predict_layer}, "
        f"dim of middle predict layer: {dim_predict_mid}, type of activate function: {activate_type}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@qDKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def qdkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)

    # IPS
    use_sample_weight = local_params["use_sample_weight"]
    sample_weight_method = local_params["sample_weight_method"]
    IPS_min = local_params["IPS_min"]
    IPS_his_seq_len = local_params['IPS_his_seq_len']

    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["IPS_min"] = IPS_min
    global_params["IPS_his_seq_len"] = IPS_his_seq_len

    global_objects["logger"].info(
        f"IPS params\n    "
        f"use IPS: {use_sample_weight}, IPS_min: {IPS_min}, IPS_his_seq_len: {IPS_his_seq_len}"
    )

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    instance_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-instance_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_duo_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    duo_cl_general_config(local_params, global_params)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-duo_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_cluster_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    cluster_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-cluster_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-ME-ADA@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_meta_optimize_cl_config(local_params):
    global_params = deepcopy(PARAMS2)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    meta_optimize_cl_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-meta_optimize_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_output_enhance_config(local_params):
    global_params = deepcopy(PARAMS2)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    output_enhance_general_config(local_params, global_params, global_objects)
    global_params["datasets_config"]["train"]["kt_output_enhance"] = {}
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-output_enhance@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_mutual_enhance4long_tail_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    mutual_enhance4long_tail_general_config(local_params, global_params, global_objects)

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-ME4long_tail@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_dro_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)
    dro_general_config(local_params, global_params, global_objects)

    data_type = global_params["datasets_config"]["data_type"]
    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    global_objects["dro"] = {
        "propensity": torch.tensor(
            cal_propensity(dataset_train, local_params["num_question"], data_type, "question")
        ).to(global_params["device"])
    }

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT-DRO@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_instance_cl_srs_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    qdkt_general_config(local_params, global_params, global_objects)

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "srs"
    datasets_train_config["srs"] = {}

    max_seq_len = local_params["max_seq_len"]
    aug_order = eval(local_params["aug_order"])
    mask_prob = local_params["mask_prob"]
    replace_prob = local_params["replace_prob"]
    crop_prob = local_params["crop_prob"]
    permute_prob = local_params["permute_prob"]
    datasets_train_config["srs"]["max_seq_len"] = max_seq_len
    datasets_train_config["srs"]["aug_order"] = aug_order
    datasets_train_config["srs"]["mask_prob"] = mask_prob
    datasets_train_config["srs"]["replace_prob"] = replace_prob
    datasets_train_config["srs"]["crop_prob"] = crop_prob
    datasets_train_config["srs"]["permute_prob"] = permute_prob

    temp = local_params["temp"]
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["other"] = {"instance_cl": {}}
    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["temp"] = temp
    global_params["loss_config"]["cl loss"] = weight_cl_loss

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@qDKT@@", "@@qDKT_instance_cl_srs@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def qdkt_core_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)

    # 配置模型参数
    global_params["models_config"] = {
        "kt_model": deepcopy(qDKT_CORE_MODEL_PARAMS)
    }
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_emb = local_params["dim_emb"]
    fusion_mode = local_params["fusion_mode"]
    dropout = local_params["dropout"]

    # embed layer
    embed_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    embed_config["concept"] = [num_concept, dim_emb]
    embed_config["question"] = [num_question, dim_emb]

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]
    encoder_config["dim_emb"] = dim_emb
    encoder_config["fusion_mode"] = fusion_mode
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}\n    "
        f"fusion_mode: {fusion_mode}, dim_emb: {dim_emb}, dropout: {dropout}"
    )

    global_params["loss_config"]["KL loss"] = 1

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]
        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@qDKT_CORE@@seed_{local_params['seed']}"
            f"@@{setting_name}@@{train_file_name.replace('.txt', '')}"
        )
        save_params(global_params, global_objects)

    return global_params, global_objects
