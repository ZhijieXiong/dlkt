def dro_general_config(local_params, global_params, global_objects):
    datasets_train_config = global_params["datasets_config"]["train"]
    datasets_train_config["type"] = "srs"
    datasets_train_config["srs"] = {}

    use_dro = local_params["use_dro"]
    beta = local_params["beta"]
    alpha = local_params["alpha"]
    max_seq_len = local_params["max_seq_len"]

    datasets_train_config["srs"]["max_seq_len"] = max_seq_len
