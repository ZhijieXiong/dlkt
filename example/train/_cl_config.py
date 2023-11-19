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
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["informative_aug"]["insert_prob"] = insert_prob
        datasets_train_config["kt4aug"]["informative_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["informative_aug"]["num_concept"] = local_params["num_concept"]
        datasets_train_config["kt4aug"]["informative_aug"]["num_question"] = local_params["num_question"]
        datasets_train_config["kt4aug"]["informative_aug"]["offline_sim_type"] = local_params["offline_sim_type"]
    else:
        raise NotImplementedError()

    # instance CL参数
    instance_cl_config = global_params["other"]["instance_cl"]
    instance_cl_config["temp"] = local_params["temp"]
    instance_cl_config["use_warm_up4cl"] = local_params["use_warm_up4cl"]
    instance_cl_config["epoch_warm_up4cl"] = local_params["epoch_warm_up4cl"]
    instance_cl_config["use_online_sim"] = local_params["use_online_sim"]
    instance_cl_config["use_warm_up4online_sim"] = local_params["use_warm_up4online_sim"]
    instance_cl_config["epoch_warm_up4online_sim"] = local_params["epoch_warm_up4online_sim"]
    instance_cl_config["cl_type"] = local_params["cl_type"]

    # max entropy adv aug参数
    max_entropy_aug_config = global_params["other"]["max_entropy_aug"]
    max_entropy_aug_config["use_adv_aug"] = local_params["use_adv_aug"]
    max_entropy_aug_config["epoch_interval_generate"] = local_params["epoch_interval_generate"]
    max_entropy_aug_config["loop_adv"] = local_params["loop_adv"]
    max_entropy_aug_config["epoch_generate"] = local_params["epoch_generate"]
    max_entropy_aug_config["adv_learning_rate"] = local_params["adv_learning_rate"]
    max_entropy_aug_config["eta"] = local_params["eta"]
    max_entropy_aug_config["gamma"] = local_params["gamma"]

    # 损失权重
    global_params["loss_config"]["cl loss"] = local_params["weight_cl_loss"]

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, data_type)


def duo_cl_config(local_params, global_params):
    # 配置数据集参数
    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "kt4aug"
    datasets_train_config["kt4aug"]["aug_type"] = "semantic_aug"
    datasets_train_config["kt4aug"]["num_aug"] = 1

    # 损失权重
    global_params["loss_config"]["cl loss"] = local_params["weight_cl_loss"]

    # 对比学习温度系数
    global_params["other"]["duo_cl"]["temp"] = local_params["temp"]


def cluster_cl_config(local_params, global_params, global_objects):
    # 配置数据集参数
    aug_type = local_params["aug_type"]
    mask_prob = local_params["mask_prob"]
    crop_prob = local_params["crop_prob"]
    insert_prob = local_params["insert_prob"]
    permute_prob = local_params["permute_prob"]
    replace_prob = local_params["replace_prob"]
    hard_neg_prob = local_params["hard_neg_prob"]
    aug_order = eval(local_params["aug_order"])
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
    elif aug_type == "informative_aug":
        datasets_train_config["kt4aug"]["informative_aug"]["aug_order"] = aug_order
        datasets_train_config["kt4aug"]["informative_aug"]["mask_prob"] = mask_prob
        datasets_train_config["kt4aug"]["informative_aug"]["crop_prob"] = crop_prob
        datasets_train_config["kt4aug"]["informative_aug"]["insert_prob"] = insert_prob
        datasets_train_config["kt4aug"]["informative_aug"]["replace_prob"] = replace_prob
        datasets_train_config["kt4aug"]["informative_aug"]["num_concept"] = local_params["num_concept"]
        datasets_train_config["kt4aug"]["informative_aug"]["num_question"] = local_params["num_question"]
        datasets_train_config["kt4aug"]["informative_aug"]["offline_sim_type"] = local_params["offline_sim_type"]
    else:
        raise NotImplementedError()

    # cluster CL参数
    instance_cl_config = global_params["other"]["cluster_cl"]
    instance_cl_config["temp"] = local_params["temp"]
    instance_cl_config["use_warm_up4cl"] = local_params["use_warm_up4cl"]
    instance_cl_config["epoch_warm_up4cl"] = local_params["epoch_warm_up4cl"]
    instance_cl_config["use_online_sim"] = local_params["use_online_sim"]
    instance_cl_config["use_warm_up4online_sim"] = local_params["use_warm_up4online_sim"]
    instance_cl_config["epoch_warm_up4online_sim"] = local_params["epoch_warm_up4online_sim"]
    instance_cl_config["num_cluster"] = local_params["num_cluster"]

    # max entropy adv aug参数
    max_entropy_aug_config = global_params["other"]["max_entropy_aug"]
    max_entropy_aug_config["use_adv_aug"] = local_params["use_adv_aug"]
    max_entropy_aug_config["epoch_interval_generate"] = local_params["epoch_interval_generate"]
    max_entropy_aug_config["loop_adv"] = local_params["loop_adv"]
    max_entropy_aug_config["epoch_generate"] = local_params["epoch_generate"]
    max_entropy_aug_config["adv_learning_rate"] = local_params["adv_learning_rate"]
    max_entropy_aug_config["eta"] = local_params["eta"]
    max_entropy_aug_config["gamma"] = local_params["gamma"]

    # 损失权重
    global_params["loss_config"]["cl loss"] = local_params["weight_cl_loss"]

    # Q_table
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    global_objects["data"]["Q_table"] = global_objects["file_manager"].get_q_table(dataset_name, data_type)
