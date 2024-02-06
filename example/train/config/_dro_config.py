def dro_general_config(local_params, global_params, global_objects):
    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "srs"

    use_dro = local_params["use_dro"]
