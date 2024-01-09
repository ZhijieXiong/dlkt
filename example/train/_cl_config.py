import json
import os
from copy import deepcopy

from lib.template.dataset_params_template import KT_RANDOM_AUG_PARAMS, KT_INFORMATIVE_AUG_PARAMS
from lib.template.model.Extractor import MODEL_PARAMS as EXTRACTOR_PARAMS
from lib.template.other_params_template import *
from lib.util.parse import get_high_dis_qc
from lib.util.data import read_preprocessed_file, load_json, write_json


def instance_cl_general_config(local_params, global_params, global_objects):
    # 配置数据集参数
    aug_type = local_params["aug_type"]
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    insert_prob = local_params["insert_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    use_hard_neg = local_params["use_hard_neg"]
    hard_neg_prob = local_params["hard_neg_prob"]
    aug_order = eval(local_params["aug_order"])
    random_select_aug_len = local_params["use_random_select_aug_len"]
    data_aug_type4cl = local_params["data_aug_type4cl"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = aug_type
    if data_aug_type4cl == "original_data_aug":
        datasets_train_config["kt4aug"]["num_aug"] = 2
    elif data_aug_type4cl == "hybrid":
        datasets_train_config["kt4aug"]["num_aug"] = 1
    else:
        datasets_train_config["kt4aug"]["num_aug"] = 0
    if aug_type == "random_aug":
        datasets_train_config["kt4aug"]["random_aug"] = deepcopy(KT_RANDOM_AUG_PARAMS)
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["use_hard_neg"] = use_hard_neg
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"] = deepcopy(KT_INFORMATIVE_AUG_PARAMS)
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
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

    # instance CL参数
    temp = local_params["temp"]
    use_warm_up4cl = local_params["use_warm_up4cl"]
    epoch_warm_up4cl = local_params["epoch_warm_up4cl"]
    use_weight_dynamic = local_params["use_weight_dynamic"]
    weight_dynamic_type = local_params["weight_dynamic_type"]
    multi_step_weight = eval(local_params["multi_step_weight"])
    linear_increase_epoch = local_params["linear_increase_epoch"]
    linear_increase_value = local_params["linear_increase_value"]
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    latent_type4cl = local_params["latent_type4cl"]
    use_emb_dropout4cl = local_params["use_emb_dropout4cl"]
    emb_dropout4cl = local_params["emb_dropout4cl"]
    use_neg = local_params["use_neg"]
    use_neg_filter = local_params["use_neg_filter"]
    neg_sim_threshold = local_params["neg_sim_threshold"]
    use_stop_cl_after = local_params["use_stop_cl_after"]
    epoch_stop_cl = local_params["epoch_stop_cl"]
    cl_space = local_params["cl_space"]
    num2drop_question4dis = local_params.get("num2drop_question4dis", 50)
    num2drop_concept4dis = local_params.get("num2drop_concept4dis", 500)
    min_seq_len4dis = local_params.get("min_seq_len4dis", 30)
    dis_threshold = local_params.get("dis_threshold", 0.2)

    global_params["other"]["instance_cl"] = {}
    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["epoch_warm_up4cl"] = epoch_warm_up4cl
    instance_cl_config["use_warm_up4cl"] = use_warm_up4cl
    instance_cl_config["temp"] = temp
    instance_cl_config["use_online_sim"] = use_online_sim
    instance_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    instance_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    instance_cl_config["latent_type4cl"] = latent_type4cl
    instance_cl_config["random_select_aug_len"] = random_select_aug_len
    instance_cl_config["use_weight_dynamic"] = use_weight_dynamic
    instance_cl_config["weight_dynamic"] = {}
    instance_cl_config["weight_dynamic"][weight_dynamic_type] = {}
    instance_cl_config["weight_dynamic"]["type"] = weight_dynamic_type
    if weight_dynamic_type == "multi_step":
        instance_cl_config["weight_dynamic"]["multi_step"]["step_weight"] = multi_step_weight
    elif weight_dynamic_type == "linear_increase":
        instance_cl_config["weight_dynamic"]["linear_increase"] = {}
        instance_cl_config["weight_dynamic"]["linear_increase"]["epoch"] = linear_increase_epoch
        instance_cl_config["weight_dynamic"]["linear_increase"]["value"] = linear_increase_value
    else:
        raise NotImplementedError()
    instance_cl_config["use_emb_dropout4cl"] = use_emb_dropout4cl
    instance_cl_config["emb_dropout4cl"] = emb_dropout4cl
    instance_cl_config["data_aug_type4cl"] = data_aug_type4cl
    weight_dynamic = instance_cl_config["weight_dynamic"][weight_dynamic_type]
    instance_cl_config["use_neg"] = use_neg
    instance_cl_config["use_neg_filter"] = use_neg_filter
    instance_cl_config["neg_sim_threshold"] = neg_sim_threshold
    instance_cl_config["use_stop_cl_after"] = use_stop_cl_after
    instance_cl_config["epoch_stop_cl"] = epoch_stop_cl
    instance_cl_config["cl_space"] = cl_space

    # max entropy adv aug参数
    use_adv_aug = local_params["use_adv_aug"]
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]

    global_params["other"]["max_entropy_adv_aug"] = deepcopy(MAX_ENTROPY_ADV_AUG)
    max_entropy_aug_config = global_params["other"]["max_entropy_adv_aug"]
    instance_cl_config["use_adv_aug"] = use_adv_aug
    if use_adv_aug:
        max_entropy_aug_config["epoch_interval_generate"] = epoch_interval_generate
        max_entropy_aug_config["loop_adv"] = loop_adv
        max_entropy_aug_config["epoch_generate"] = epoch_generate
        max_entropy_aug_config["adv_learning_rate"] = adv_learning_rate
        max_entropy_aug_config["eta"] = eta
        max_entropy_aug_config["gamma"] = gamma

    # 在output space空间中做对比学习需要的数据
    if cl_space == "output":
        dataset_config_this = global_params["datasets_config"][global_params["datasets_config"]["dataset_this"]]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
        high_distinction_q_path = os.path.join(setting_dir, file_name.replace(".txt", "_high_distinction_question.json"))

        if os.path.exists(high_distinction_q_path):
            high_distinction_q = load_json(high_distinction_q_path)
            global_objects["data"]["high_distinction_q"] = high_distinction_q
        else:
            dataset_train = read_preprocessed_file(os.path.join(
                global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
                global_params["datasets_config"]["train"]["file_name"]
            ))
            get_high_dis_qc_params = {
                "num2drop4question": num2drop_question4dis,
                "num2drop4concept": num2drop_concept4dis,
                "min_seq_len": min_seq_len4dis,
                "dis_threshold": dis_threshold
            }
            high_distinction_c, high_distinction_q = get_high_dis_qc(dataset_train,
                                                                     global_params["datasets_config"]["data_type"],
                                                                     get_high_dis_qc_params)
            global_objects["data"]["high_distinction_q"] = high_distinction_q
            write_json(high_distinction_q, high_distinction_q_path)
        data_type = global_params["datasets_config"]["data_type"]

        if data_type == "single_concept":
            high_distinction_c_path = os.path.join(setting_dir, file_name.replace(".txt", "_high_distinction_concept.json"))
            if os.path.exists(high_distinction_c_path):
                high_distinction_c = load_json(high_distinction_c_path)
                global_objects["data"]["high_distinction_c"] = high_distinction_c
            else:
                global_objects["data"]["high_distinction_c"] = list(map(
                    lambda q_id: global_objects["data"]["question2concept"][q_id][0],
                    high_distinction_q
                ))
                write_json(global_objects["data"]["high_distinction_c"], high_distinction_c_path)

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss

    # 打印参数
    info_aug_params_str = f"offline sim type: {local_params['offline_sim_type']}, use online sim: {use_online_sim}, " \
                          f"use warm up for online sim: {use_warm_up4online_sim}, num of warm up epoch for online sim: {epoch_warm_up4online_sim}"
    global_objects["logger"].info(
        f"input data aug\n"
        f"    aug_type: {aug_type}, aug order: {local_params['aug_order']}, use hard neg: {use_hard_neg}, random aug len: {random_select_aug_len}\n"
        f"    {'use random data aug' if aug_type == 'random_aug' else f'use info data aug, {info_aug_params_str}'}\n"
        f"    mask prob: {mask_prob}, crop prob: {crop_prob}, replace prob: {replace_prob}, insert prob: {insert_prob}, permute prob: {permute_prob}\n"
        f"instance cl\n"
        f"    cl space: {cl_space}, temp: {temp}, weight of cl loss: {weight_cl_loss}\n"
        f"    use dynamic weight: {use_weight_dynamic}{f', dynamic weight type: {weight_dynamic_type}, {weight_dynamic_type}: {json.dumps(weight_dynamic)}' if use_weight_dynamic else ''}\n"
        f"    use warm up for cl: {use_warm_up4cl}{f', num of warm up epoch for cl: {epoch_warm_up4cl}' if use_warm_up4cl else ''}, "
        f"use stop cl after specified epoch: {use_stop_cl_after}{f', num of epoch to stop cl: {epoch_stop_cl}' if use_stop_cl_after else ''}\n"
        f"    data aug type: {data_aug_type4cl}, latent type: {latent_type4cl}, use emb dropout: {use_emb_dropout4cl}{f', emb dropout: {emb_dropout4cl}' if use_emb_dropout4cl else ''}\n"
        f"    use neg sample: {use_neg}, use neg sample filter: {use_neg_filter}{f', threshold of neg sample filter (similarity): {neg_sim_threshold}' if use_neg_filter else ''}\n"
        f"max_entropy_adv_aug\n"
        f"    use max entropy adv aug: {use_adv_aug}, interval epoch of generation: {epoch_interval_generate}, generate loops: {loop_adv}, num of generation epoch: {epoch_generate}\n"
        f"    adv lr: {adv_learning_rate}, eta: {eta}, gamma: {gamma}"
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
    # 配置数据集参数
    aug_type = local_params["aug_type"]
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    insert_prob = local_params["insert_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    hard_neg_prob = local_params["hard_neg_prob"]
    aug_order = eval(local_params["aug_order"])
    random_select_aug_len = local_params["use_random_select_aug_len"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = aug_type
    datasets_train_config["kt4aug"]["num_aug"] = 2
    if aug_type == "random_aug":
        datasets_train_config["kt4aug"]["random_aug"] = deepcopy(KT_RANDOM_AUG_PARAMS)
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"] = deepcopy(KT_INFORMATIVE_AUG_PARAMS)
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["informative_aug"]["insert_prob"] = insert_prob
        datasets_train_config["kt4aug"]["informative_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["informative_aug"]["num_concept"] = local_params["num_concept"]
        datasets_train_config["kt4aug"]["informative_aug"]["num_question"] = local_params["num_question"]
        datasets_train_config["kt4aug"]["informative_aug"]["offline_sim_type"] = local_params["offline_sim_type"]
        datasets_train_config["kt4aug"]["informative_aug"]["random_select_aug_len"] = random_select_aug_len
    else:
        raise NotImplementedError()

    # cluster CL参数
    use_warm_up4cl = local_params["use_warm_up4cl"]
    epoch_warm_up4cl = local_params["epoch_warm_up4cl"]
    temp = local_params["temp"]
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    num_cluster = local_params["num_cluster"]
    cl_type = local_params["cl_type"]

    global_params["other"]["cluster_cl"] = deepcopy(CLUSTER_CL_PARAMS)
    cluster_cl_config = global_params["other"]["cluster_cl"]
    cluster_cl_config["use_warm_up4cl"] = use_warm_up4cl
    cluster_cl_config["epoch_warm_up4cl"] = epoch_warm_up4cl
    cluster_cl_config["temp"] = temp
    cluster_cl_config["use_online_sim"] = use_online_sim
    cluster_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    cluster_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    cluster_cl_config["num_cluster"] = num_cluster
    cluster_cl_config["cl_type"] = cl_type
    cluster_cl_config["random_select_aug_len"] = random_select_aug_len

    # max entropy adv aug参数
    use_adv_aug = local_params["use_adv_aug"]
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]

    global_params["other"]["max_entropy_adv_aug"] = deepcopy(MAX_ENTROPY_ADV_AUG)
    max_entropy_aug_config = global_params["other"]["max_entropy_adv_aug"]
    cluster_cl_config["use_adv_aug"] = use_adv_aug
    if use_adv_aug:
        max_entropy_aug_config["epoch_interval_generate"] = epoch_interval_generate
        max_entropy_aug_config["loop_adv"] = loop_adv
        max_entropy_aug_config["epoch_generate"] = epoch_generate
        max_entropy_aug_config["adv_learning_rate"] = adv_learning_rate
        max_entropy_aug_config["eta"] = eta
        max_entropy_aug_config["gamma"] = gamma

    # 损失权重
    weight_cl_loss = local_params["weight_cl_loss"]
    global_params["loss_config"]["cl loss"] = weight_cl_loss

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, data_type)


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
    # global_params["grad_clip_config"]["extractor0"] = deepcopy(global_params["grad_clip_config"]["kt_model"])
    # global_params["grad_clip_config"]["extractor1"] = deepcopy(global_params["grad_clip_config"]["kt_model"])

    # 配置数据集参数
    aug_type = local_params["aug_type"]
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    insert_prob = local_params["insert_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    use_hard_neg = local_params["use_hard_neg"]
    hard_neg_prob = local_params["hard_neg_prob"]
    aug_order = eval(local_params["aug_order"])
    random_select_aug_len = local_params["use_random_select_aug_len"]

    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = aug_type
    datasets_train_config["kt4aug"]["num_aug"] = 2
    if aug_type == "random_aug":
        datasets_train_config["kt4aug"]["random_aug"] = deepcopy(KT_RANDOM_AUG_PARAMS)
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["use_hard_neg"] = use_hard_neg
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"] = deepcopy(KT_INFORMATIVE_AUG_PARAMS)
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
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

    # meta CL参数
    use_regularization = local_params["use_regularization"]
    global_params["other"]["meta_cl"] = deepcopy(META_OPTIMIZE_CL_PARAMS)
    global_params["other"]["meta_cl"]["use_regularization"] = use_regularization

    # instance CL参数
    temp = local_params["temp"]
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    cl_type = local_params["cl_type"]

    global_params["other"]["instance_cl"] = deepcopy(INSTANCE_CL_PARAMS)
    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["temp"] = temp
    instance_cl_config["use_online_sim"] = use_online_sim
    instance_cl_config["use_warm_up4online_sim"] = use_warm_up4online_sim
    instance_cl_config["epoch_warm_up4online_sim"] = epoch_warm_up4online_sim
    instance_cl_config["cl_type"] = cl_type

    # max entropy adv aug参数
    use_adv_aug = local_params["use_adv_aug"]
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]

    global_params["other"]["max_entropy_adv_aug"] = deepcopy(MAX_ENTROPY_ADV_AUG)
    max_entropy_aug_config = global_params["other"]["max_entropy_adv_aug"]
    instance_cl_config["use_adv_aug"] = use_adv_aug
    if use_adv_aug:
        max_entropy_aug_config["epoch_interval_generate"] = epoch_interval_generate
        max_entropy_aug_config["loop_adv"] = loop_adv
        max_entropy_aug_config["epoch_generate"] = epoch_generate
        max_entropy_aug_config["adv_learning_rate"] = adv_learning_rate
        max_entropy_aug_config["eta"] = eta
        max_entropy_aug_config["gamma"] = gamma

    # 损失权重
    weight_lambda = local_params["weight_lambda"]
    weight_beta = local_params["weight_beta"]
    weight_gamma = local_params["weight_gamma"]

    global_params["loss_config"]["cl loss1"] = weight_lambda
    global_params["loss_config"]["cl loss2"] = weight_beta
    global_params["loss_config"]["reg loss"] = weight_gamma

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, data_type)
