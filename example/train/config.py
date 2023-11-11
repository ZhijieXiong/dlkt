import json
import sys
import os
import inspect
from copy import deepcopy

import torch.cuda

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])


from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.util.FileManager import FileManager


def general_config(local_params, global_params, global_objects):
    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager
    global_params["save_model"] = local_params["save_model"]
    global_params["train_strategy"]["type"] = local_params["train_strategy"]
    global_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # global_params["device"] = "cpu"

    # 训练策略配置
    num_epoch = local_params["num_epoch"]
    use_early_stop = local_params["use_early_stop"]
    epoch_early_stop = local_params["epoch_early_stop"]
    use_last_average = local_params["use_last_average"]
    epoch_last_average = local_params["epoch_last_average"]
    main_metric = local_params["main_metric"]
    use_multi_metrics = local_params["use_multi_metrics"]
    mutil_metrics = local_params["multi_metrics"]

    train_strategy = global_params["train_strategy"]
    train_strategy["num_epoch"] = num_epoch
    train_strategy["main_metric"] = main_metric
    train_strategy["use_multi_metrics"] = use_multi_metrics
    if use_multi_metrics:
        train_strategy["multi_metrics"] = eval(mutil_metrics)
    if train_strategy["type"] == "valid_test":
        train_strategy["valid_test"]["use_early_stop"] = use_early_stop
        if use_early_stop:
            train_strategy["valid_test"]["epoch_early_stop"] = epoch_early_stop
    else:
        train_strategy["no_valid"]["use_average"] = use_last_average
        if use_early_stop:
            train_strategy["no_valid"]["use_average"] = epoch_last_average

    # 数据集配置
    setting_name = local_params["setting_name"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]

    datasets_config = global_params["datasets_config"]
    datasets_config["train"]["setting_name"] = setting_name
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["valid"]["setting_name"] = setting_name
    if train_strategy["type"] == "valid_test":
        datasets_config["valid"]["file_name"] = valid_file_name
    datasets_config["train"]["file_name"] = train_file_name
    datasets_config["test"]["file_name"] = test_file_name

    # 优化器配置
    kt_optimizer_type = local_params["optimizer_type"]
    kt_weight_decay = local_params["weight_decay"]
    kt_momentum = local_params["momentum"]
    kt_learning_rate = local_params["learning_rate"]
    kt_enable_lr_schedule = local_params["enable_lr_schedule"]
    kt_lr_schedule_type = local_params["lr_schedule_type"]
    kt_lr_schedule_step = local_params["lr_schedule_step"]
    kt_lr_schedule_gamma = local_params["lr_schedule_gamma"]
    kt_enable_clip_grad = local_params["enable_clip_grad"]
    kt_grad_clipped = local_params["grad_clipped"]

    optimizer_config = global_params["optimizers_config"]["kt_model"]
    optimizer_config["type"] = kt_optimizer_type
    optimizer_config[kt_optimizer_type]["lr"] = kt_learning_rate
    optimizer_config[kt_optimizer_type]["weight_decay"] = kt_weight_decay
    if kt_optimizer_type == "sgd":
        optimizer_config[kt_optimizer_type]["momentum"] = kt_momentum

    scheduler_config = global_params["schedulers_config"]["kt_model"]
    if kt_enable_lr_schedule:
        scheduler_config["use_scheduler"] = True
        scheduler_config["type"] = kt_lr_schedule_type
        if kt_lr_schedule_type == "StepLR":
            scheduler_config[kt_lr_schedule_type]["step_size"] = kt_lr_schedule_step
            scheduler_config[kt_lr_schedule_type]["gamma"] = kt_lr_schedule_gamma
        else:
            raise NotImplementedError()
    else:
        scheduler_config["use_scheduler"] = False

    grad_clip_config = global_params["grad_clip_config"]["kt_model"]
    grad_clip_config["use_clip"] = kt_enable_clip_grad
    if kt_enable_clip_grad:
        grad_clip_config["grad_clipped"] = kt_grad_clipped


def dkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

    # 配置模型参数
    num_concept = local_params["num_concept"]
    dim_emb = local_params["dim_emb"]
    dim_latent = local_params["dim_latent"]
    rnn_type = local_params["rnn_type"]
    num_rnn_layer = local_params["num_rnn_layer"]
    dropout = local_params["dropout"]
    num_predict_layer = local_params["num_predict_layer"]
    dim_predict_mid = local_params["dim_predict_mid"]
    activate_type = local_params["activate_type"]

    # embed layer
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["interaction"] = [num_concept * 2, dim_emb]

    # encoder layer
    dkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
    dkt_encoder_layer_config["dim_emb"] = dim_emb
    dkt_encoder_layer_config["dim_latent"] = dim_latent
    dkt_encoder_layer_config["rnn_type"] = rnn_type
    dkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = num_concept

    return global_params, global_objects


def qdkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

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
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["concept"] = [num_concept, dim_concept]
    kt_embed_layer_config["question"] = [num_question, dim_question]

    # encoder layer
    qdkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    qdkt_encoder_layer_config["dim_concept"] = dim_concept
    qdkt_encoder_layer_config["dim_question"] = dim_question
    qdkt_encoder_layer_config["dim_correct"] = dim_correct
    qdkt_encoder_layer_config["dim_latent"] = dim_latent
    qdkt_encoder_layer_config["rnn_type"] = rnn_type
    qdkt_encoder_layer_config["num_rnn_layer"] = num_rnn_layer

    # predict layer
    predict_layer_config = global_params["models_config"]["kt_model"]["predict_layer"]
    predict_layer_config["type"] = "direct"
    predict_layer_config["direct"]["dropout"] = dropout
    predict_layer_config["direct"]["num_predict_layer"] = num_predict_layer
    predict_layer_config["direct"]["dim_predict_in"] = dim_latent + dim_concept + dim_question
    predict_layer_config["direct"]["dim_predict_mid"] = dim_predict_mid
    predict_layer_config["direct"]["activate_type"] = activate_type
    predict_layer_config["direct"]["dim_predict_out"] = 1

    return global_params, global_objects


def akt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)

    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "AKT"

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_model = local_params["dim_model"]
    key_query_same = local_params["key_query_same"]
    num_head = local_params["num_head"]
    num_block = local_params["num_block"]
    dim_ff = local_params["dim_ff"]
    dim_final_fc = local_params["dim_final_fc"]
    separate_qa = local_params["separate_qa"]
    dropout = local_params["dropout"]

    # encoder layer
    akt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
    akt_encoder_layer_config["num_concept"] = num_concept
    akt_encoder_layer_config["num_question"] = num_question
    akt_encoder_layer_config["dim_model"] = dim_model
    akt_encoder_layer_config["key_query_same"] = key_query_same
    akt_encoder_layer_config["num_head"] = num_head
    akt_encoder_layer_config["num_block"] = num_block
    akt_encoder_layer_config["dim_ff"] = dim_ff
    akt_encoder_layer_config["dim_final_fc"] = dim_final_fc
    akt_encoder_layer_config["separate_qa"] = separate_qa
    akt_encoder_layer_config["dropout"] = dropout

    # 损失权重
    global_params["loss_config"]["rasch_loss"] = local_params["weight_rasch_loss"]

    return global_params, global_objects
