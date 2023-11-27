def instance_cl_general_config(local_params, global_params, global_objects):
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
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
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

    # instance CL参数
    temp = local_params["temp"]
    use_warm_up4cl = local_params["use_warm_up4cl"]
    epoch_warm_up4cl = local_params["epoch_warm_up4cl"]
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    cl_type = local_params["cl_type"]

    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["temp"] = temp
    instance_cl_config["use_warm_up4cl"] = use_warm_up4cl
    instance_cl_config["epoch_warm_up4cl"] = epoch_warm_up4cl
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

    max_entropy_aug_config = global_params["other"]["max_entropy_aug"]
    max_entropy_aug_config["use_adv_aug"] = use_adv_aug
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

    aug_table = {
        "mask": mask_prob,
        "crop": crop_prob,
        "replace": replace_prob,
        "insert": insert_prob,
        "permute": permute_prob
    }
    if aug_type == "random_aug":
        params_str = "random_aug"
    elif aug_type == "informative_aug":
        params_str = f"informative_aug"
    else:
        raise NotImplementedError()
    if random_select_aug_len:
        params_str += "-random_seq_len"
    else:
        params_str += "-original_seq_len"

    for aug in aug_order:
        params_str += f"-{aug}-{aug_table[aug]}"

    if local_params["use_adv_aug"]:
        params_str += f"-adv_aug-{epoch_interval_generate}-{loop_adv}-{epoch_generate}-{adv_learning_rate}-{eta}-{gamma}"

    params_str += f"@@temp_{temp}-wight_cl-{weight_cl_loss}"

    return params_str


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

    params_str = f"temp_{temp}-wight_cl-{weight_cl_loss}"

    return params_str


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
        datasets_train_config["kt4aug"]["random_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["random_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["random_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["random_aug"]["permute_prob"] = permute_prob
        datasets_train_config["kt4aug"]["random_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["random_aug"]["hard_neg_prob"] = hard_neg_prob
        datasets_train_config["kt4aug"]["random_aug"]["random_select_aug_len"] = random_select_aug_len
    elif aug_type == "informative_aug":
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
    temp = local_params["temp"]
    use_warm_up4cl = local_params["use_warm_up4cl"]
    epoch_warm_up4cl = local_params["epoch_warm_up4cl"]
    use_online_sim = local_params["use_online_sim"]
    use_warm_up4online_sim = local_params["use_warm_up4online_sim"]
    epoch_warm_up4online_sim = local_params["epoch_warm_up4online_sim"]
    num_cluster = local_params["num_cluster"]
    cl_type = local_params["cl_type"]

    cluster_cl_config = global_params["other"]["cluster_cl"]
    cluster_cl_config["temp"] = temp
    cluster_cl_config["use_warm_up4cl"] = use_warm_up4cl
    cluster_cl_config["epoch_warm_up4cl"] = epoch_warm_up4cl
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

    max_entropy_aug_config = global_params["other"]["max_entropy_aug"]
    max_entropy_aug_config["use_adv_aug"] = use_adv_aug
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

    aug_table = {
        "mask": mask_prob,
        "crop": crop_prob,
        "replace": replace_prob,
        "insert": insert_prob,
        "permute": permute_prob
    }
    if aug_type == "random_aug":
        params_str = "random_aug"
    elif aug_type == "informative_aug":
        params_str = f"informative_aug"
    else:
        raise NotImplementedError()
    if random_select_aug_len:
        params_str += "-random_seq_len"
    else:
        params_str += "-original_seq_len"

    for aug in aug_order:
        params_str += f"-{aug}-{aug_table[aug]}"

    if local_params["use_adv_aug"]:
        params_str += f"-adv_aug-{epoch_interval_generate}-{loop_adv}-{epoch_generate}-{adv_learning_rate}-{eta}-{gamma}"

    params_str += f"@@temp_{temp}-wight_cl-{weight_cl_loss}"

    return params_str
