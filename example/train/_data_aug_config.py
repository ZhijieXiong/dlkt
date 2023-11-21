def max_entropy_adv_aug_general_config(local_params, global_params):
    max_entropy_adv_aug_config = global_params["other"]["max_entropy_adv_aug"]
    max_entropy_adv_aug_config["use_warm_up"] = local_params["use_warm_up"]
    max_entropy_adv_aug_config["epoch_warm_up"] = local_params["epoch_warm_up"]
    max_entropy_adv_aug_config["epoch_interval_generate"] = local_params["epoch_interval_generate"]
    max_entropy_adv_aug_config["loop_adv"] = local_params["loop_adv"]
    max_entropy_adv_aug_config["epoch_generate"] = local_params["epoch_generate"]
    max_entropy_adv_aug_config["adv_learning_rate"] = local_params["adv_learning_rate"]
    max_entropy_adv_aug_config["eta"] = local_params["eta"]
    max_entropy_adv_aug_config["gamma"] = local_params["gamma"]
    # 对抗数据的损失权重
    global_params["loss_config"]["adv predict loss"] = local_params["weight_adv_pred_loss"]
