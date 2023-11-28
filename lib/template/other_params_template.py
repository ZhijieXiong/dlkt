DUO_CL_PARAMS = {
    "temp": 0.05,
    "cl_type": "last_time",
}


INSTANCE_CL_PARAMS = {
    "temp": 0.05,
    "use_warm_up4cl": False,
    "epoch_warm_up4cl": 4,
    "use_online_sim": True,
    "use_warm_up4online_sim": True,
    "epoch_warm_up4online_sim": 4,
    # "last_time" or "all_time" or "mean_pool"
    "cl_type": "last_time"
}


CLUSTER_CL_PARAMS = {
    "random_select_aug_len": False,
    "num_cluster": 32,
    "temp": 0.05,
    "use_warm_up4cl": False,
    "epoch_warm_up4cl": 4,
    "use_online_sim": True,
    "use_warm_up4online_sim": True,
    "epoch_warm_up4online_sim": 4,
    "cl_type": "last_time",
    "use_adv_data": False
}


META_OPTIMIZE_CL_PARAMS = {

}


MAX_ENTROPY_ADV_AUG_CL = {
    "use_adv_aug": False,
    "epoch_interval_generate": 1,
    "loop_adv": 3,
    "epoch_generate": 40,
    "adv_learning_rate": 10,
    "eta": 5,
    "gamma": 1
}


MAX_ENTROPY_ADV_AUG = {
    "use_warm_up": False,
    "epoch_warm_up": 4,
    "epoch_interval_generate": 1,
    "loop_adv": 3,
    "epoch_generate": 40,
    "adv_learning_rate": 10,
    "eta": 5,
    "gamma": 1
}

