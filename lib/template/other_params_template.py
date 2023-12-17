DUO_CL_PARAMS = {
    "temp": 0.05,
    "cl_type": "last_time",
}


INSTANCE_CL_PARAMS = {
    "temp": 0.01,
    "use_online_sim": True,
    "use_warm_up4online_sim": True,
    "epoch_warm_up4online_sim": 4,
    "random_select_aug_len": False,
    # "last_time" or "all_time" or "mean_pool"
    "latent_type4cl": "last_time",
    "use_weight_dynamic": False,
    "weight_dynamic": {
        # "multi_step", "linear_increase"
        "type": "linear_increase",
        "multi_step": {
            "step_weight": []
        },
        "linear_increase": {
            "epoch": 1,
            "value": 0.01
        }
    },
    "use_emb_dropout4cl": True,
    "emb_dropout4cl": 0.1,
    # "original_data_aug", "model_aug", "hybrid"
    "data_aug_type4cl": "hybrid",
    "use_adv_aug": False,
    "use_neg": True,
    "use_neg_filter": True,
    "neg_sim_threshold": 0.5,
    "use_stop_cl_after": True,
    "epoch_stop_cl": 3
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
    "use_anneal": False,
    "ablation": {
        "use_vae": True
    }
}
