DUO_CL_PARAMS = {
    "temp": 0.05,
    "cl_type": "last_time",
}


INSTANCE_CL_PARAMS = {
    "use_adv_aug": False,
    "random_select_aug_len": False,
    "temp": 0.05,
    "use_online_sim": True,
    "use_warm_up4online_sim": True,
    "epoch_warm_up4online_sim": 4,
    # "last_time" or "all_time" or "mean_pool"
    "cl_type": "last_time",
    "akt_seq_representation": "encoder_output"
}


CLUSTER_CL_PARAMS = {
    "use_adv_aug": False,
    "random_select_aug_len": False,
    "num_cluster": 32,
    "temp": 0.05,
    "use_warm_up4cluster_cl": True,
    "epoch_warm_up4cluster_cl": 4,
    "use_online_sim": True,
    "use_warm_up4online_sim": True,
    "epoch_warm_up4online_sim": 4,
    # "last_time" or "all_time"
    "cl_type": "last_time"
}


META_OPTIMIZE_CL_PARAMS = {
    "use_regularization": True
}


MAX_ENTROPY_ADV_AUG = {
    "epoch_interval_generate": 1,
    "loop_adv": 3,
    "epoch_generate": 40,
    "adv_learning_rate": 10,
    "eta": 5,
    "gamma": 1
}


AC_VAE_PARAMS = {
    "use_anneal": False
}
