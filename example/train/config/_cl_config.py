from copy import deepcopy
from lib.template.kt_model.Extractor import MODEL_PARAMS as EXTRACTOR_PARAMS


def aug_general_config(local_params, global_params, global_objects):
    aug_type = local_params["aug_type"]
    aug_order = eval(local_params["aug_order"])
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    insert_prob = local_params["insert_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    # 下面4个参数有些数据增强可能不考虑
    use_hard_neg = local_params.get("use_hard_neg", False)
    hard_neg_prob = local_params.get("hard_neg_prob", 1)
    random_select_aug_len = local_params.get("use_random_select_aug_len", False)
    data_aug_type4cl = local_params.get("data_aug_type4cl", "original_data_aug")

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"] = {}
    datasets_train_config["kt4aug"]["aug_type"] = aug_type
    if data_aug_type4cl == "original_data_aug":
        datasets_train_config["kt4aug"]["num_aug"] = 2
    elif data_aug_type4cl == "hybrid":
        datasets_train_config["kt4aug"]["num_aug"] = 1
    else:
        datasets_train_config["kt4aug"]["num_aug"] = 0
    if aug_type == "random_aug":
        datasets_train_config["kt4aug"]["random_aug"] = {}
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["use_hard_neg"] = use_hard_neg
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"] = {}
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["informative_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["informative_aug"]["insert_prob"] = insert_prob
        datasets_train_config["kt4aug"]["informative_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["informative_aug"]["use_hard_neg"] = use_hard_neg
        datasets_train_config["kt4aug"]["informative_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["informative_aug"]["num_concept"] = local_params["num_concept"]
        datasets_train_config["kt4aug"]["informative_aug"]["num_question"] = local_params["num_question"]
        datasets_train_config["kt4aug"]["informative_aug"]["offline_sim_type"] = local_params["offline_sim_type"]
        datasets_train_config["kt4aug"]["informative_aug"]["random_select_aug_len"] = random_select_aug_len
    else:
        raise NotImplementedError()

    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    info_aug_params_str = f"offline sim type: {local_params['offline_sim_type']}, use online sim: {use_online_sim}, " \
                          f"use warm up for online sim: {use_warm_up4online_sim}, num of warm up epoch for online sim: {epoch_warm_up4online_sim}"
    global_objects["logger"].info(
        f"input data aug\n    "
        f"aug_type: {aug_type}, aug_order: {local_params['aug_order']}, use_hard_neg: {use_hard_neg}, "
        f"random_select_aug_len: {random_select_aug_len}\n     "
        f"{'use random data aug' if aug_type == 'random_aug' else f'use info data aug, {info_aug_params_str}'}\n    "
        f"mask_prob: {mask_prob}, crop_prob: {crop_prob}, replace_prob: {replace_prob}, insert_prob: {insert_prob}, "
        f"permute_prob: {permute_prob}"
    )


def instance_cl_general_config(local_params, global_params, global_objects):
    # 配置数据集参数
    random_select_aug_len = local_params["use_random_select_aug_len"]
    data_aug_type4cl = local_params["data_aug_type4cl"]

    aug_general_config(local_params, global_params, global_objects)

    # instance CL
    temp = local_params["temp"]
    latent_type4cl = local_params["latent_type4cl"]
    # info aug online sim
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    # dropout aug
    use_emb_dropout4cl = local_params["use_emb_dropout4cl"]
    emb_dropout4cl = local_params["emb_dropout4cl"]
    # neg sample config
    use_neg_filter = local_params["use_neg_filter"]
    neg_sim_threshold = local_params["neg_sim_threshold"]
    multi_stage = local_params["multi_stage"]

    global_params["other"]["instance_cl"] = {}
    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["temp"] = temp
    instance_cl_config["use_online_sim"] = use_online_sim
    instance_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    instance_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    instance_cl_config["latent_type4cl"] = latent_type4cl
    instance_cl_config["random_select_aug_len"] = random_select_aug_len
    instance_cl_config["use_emb_dropout4cl"] = use_emb_dropout4cl
    instance_cl_config["emb_dropout4cl"] = emb_dropout4cl
    instance_cl_config["data_aug_type4cl"] = data_aug_type4cl
    instance_cl_config["use_neg_filter"] = use_neg_filter
    instance_cl_config["neg_sim_threshold"] = neg_sim_threshold
    instance_cl_config["multi_stage"] = multi_stage

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss

    # 打印参数
    global_objects["logger"].info(
        f"instance cl\n    "
        f"temp: {temp}, weight_cl_loss: {weight_cl_loss}\n    "
        f"data_aug_type4cl: {data_aug_type4cl}, latent_type4cl: {latent_type4cl}, use_emb_dropout4cl: {use_emb_dropout4cl}"
        f"{f', emb_dropout4cl: {emb_dropout4cl}' if use_emb_dropout4cl else ''}\n    "
        f"use_neg_filter: {use_neg_filter}{f', neg_sim_threshold: {neg_sim_threshold}' if use_neg_filter else ''}\n"
    )


def duo_cl_general_config(local_params, global_params):
    # 配置数据集参数
    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = "semantic_aug"
    datasets_train_config["kt4aug"]["num_aug"] = 1

    # duo CL参数
    temp = local_params["temp"]
    cl_type = local_params["cl_type"]
    global_params["other"]["duo_cl"]["temp"] = temp
    global_params["other"]["duo_cl"]["cl_type"] = cl_type

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss


def cluster_cl_general_config(local_params, global_params, global_objects):
    aug_general_config(local_params, global_params, global_objects)

    # cluster CL参数
    temp = local_params["temp"]
    latent_type4cl = local_params["latent_type4cl"]
    num_cluster = local_params["num_cluster"]
    random_select_aug_len = local_params["use_random_select_aug_len"]
    # warm up for cluster CL
    use_warm_up4cl = local_params["use_warm_up4cl"]
    epoch_warm_up4cl = local_params["epoch_warm_up4cl"]
    # online sim for online similarity
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    # model aug
    data_aug_type4cl = local_params["data_aug_type4cl"]
    use_emb_dropout4cl = local_params["use_emb_dropout4cl"]
    emb_dropout4cl = local_params["emb_dropout4cl"]
    multi_stage = local_params["multi_stage"]

    global_params["other"]["cluster_cl"] = {}
    cluster_cl_config = global_params["other"]["cluster_cl"]
    cluster_cl_config["use_warm_up4cl"] = use_warm_up4cl
    cluster_cl_config["epoch_warm_up4cl"] = epoch_warm_up4cl
    cluster_cl_config["temp"] = temp
    cluster_cl_config["use_online_sim"] = use_online_sim
    cluster_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    cluster_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    cluster_cl_config["num_cluster"] = num_cluster
    cluster_cl_config["latent_type4cl"] = latent_type4cl
    cluster_cl_config["random_select_aug_len"] = random_select_aug_len
    cluster_cl_config["data_aug_type4cl"] = data_aug_type4cl
    cluster_cl_config["use_emb_dropout4cl"] = use_emb_dropout4cl
    cluster_cl_config["emb_dropout4cl"] = emb_dropout4cl
    cluster_cl_config["multi_stage"] = multi_stage

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss

    # 打印参数
    global_objects["logger"].info(
        f"cluster cl\n    "
        f"temp: {temp}, weight_cl_loss: {weight_cl_loss}, num_cluster: {num_cluster}, use_warm_up4cl: {use_warm_up4cl}"
        f"{f', epoch_warm_up4cl: {epoch_warm_up4cl}' if use_warm_up4cl else ''}, "
    )


def meta_optimize_cl_general_config(local_params, global_params, global_objects):
    # 配置Extractor参数
    extractor_layers = eval(local_params["extractor_layers"])
    extractor_active_func = local_params["extractor_active_func"]

    global_params["models_config"]["extractor"] = deepcopy(EXTRACTOR_PARAMS)
    global_params["models_config"]["extractor"]["layers"] = extractor_layers
    global_params["models_config"]["extractor"]["active_func"] = extractor_active_func

    # Extractor的优化器、学习率衰减、梯度裁剪配置和知识追踪模型一样
    global_params["optimizers_config"]["extractor0"] = deepcopy(global_params["optimizers_config"]["kt_model"])
    global_params["optimizers_config"]["extractor1"] = deepcopy(global_params["optimizers_config"]["kt_model"])
    global_params["schedulers_config"]["extractor0"] = deepcopy(global_params["schedulers_config"]["kt_model"])
    global_params["schedulers_config"]["extractor1"] = deepcopy(global_params["schedulers_config"]["kt_model"])

    aug_general_config(local_params, global_params, global_objects)

    # meta CL参数
    use_regularization = local_params["use_regularization"]
    temp = local_params["temp"]
    latent_type4cl = local_params["latent_type4cl"]
    random_select_aug_len = local_params["use_random_select_aug_len"]
    # online sim for online similarity
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    # model aug
    data_aug_type4cl = local_params["data_aug_type4cl"]
    use_emb_dropout4cl = local_params["use_emb_dropout4cl"]
    emb_dropout4cl = local_params["emb_dropout4cl"]

    global_params["other"]["meta_cl"] = {}
    meta_cl_config = global_params["other"]["meta_cl"]
    meta_cl_config["use_regularization"] = use_regularization
    meta_cl_config["temp"] = temp
    meta_cl_config["latent_type4cl"] = latent_type4cl
    meta_cl_config["random_select_aug_len"] = random_select_aug_len
    meta_cl_config["use_online_sim"] = use_online_sim
    meta_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    meta_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    meta_cl_config["data_aug_type4cl"] = data_aug_type4cl
    meta_cl_config["use_emb_dropout4cl"] = use_emb_dropout4cl
    meta_cl_config["emb_dropout4cl"] = emb_dropout4cl

    # 损失权重
    weight_lambda = local_params["weight_lambda"]
    weight_beta = local_params["weight_beta"]
    weight_gamma = local_params["weight_gamma"]

    global_params["loss_config"]["cl loss1"] = weight_lambda
    global_params["loss_config"]["cl loss2"] = weight_beta
    global_params["loss_config"]["reg loss"] = weight_gamma

    # 打印参数
    global_objects["logger"].info(
        f"meta cl\n    "
        f"temp: {temp}, use_regularization: {use_regularization}, weight_lambda: {weight_lambda}, "
        f"weight_beta: {weight_beta}, weight_gamma: {weight_gamma}"
    )
