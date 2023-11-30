def max_entropy_adv_aug_general_config(local_params, global_params):
    # adv aug相关参数
    epoch_interval_generate = local_params["epoch_interval_generate"]
    loop_adv = local_params["loop_adv"]
    epoch_generate = local_params["epoch_generate"]
    adv_learning_rate = local_params["adv_learning_rate"]
    eta = local_params["eta"]
    gamma = local_params["gamma"]
    weight_adv_pred_loss = local_params["weight_adv_pred_loss"]

    max_entropy_adv_aug_config = global_params["other"]["max_entropy_adv_aug"]
    max_entropy_adv_aug_config["epoch_interval_generate"] = epoch_interval_generate
    max_entropy_adv_aug_config["loop_adv"] = loop_adv
    max_entropy_adv_aug_config["epoch_generate"] = epoch_generate
    max_entropy_adv_aug_config["adv_learning_rate"] = adv_learning_rate
    max_entropy_adv_aug_config["eta"] = eta
    max_entropy_adv_aug_config["gamma"] = gamma
    global_params["loss_config"]["adv predict loss"] = weight_adv_pred_loss

    return (f"{epoch_interval_generate}-{loop_adv}-{epoch_generate}-{adv_learning_rate}-"
            f"{eta}-{gamma}-{weight_adv_pred_loss}")
