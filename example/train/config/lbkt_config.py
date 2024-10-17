import os.path

from ._config import *

from lib.util.basic import *
from lib.util.data import dataset_delete_pad, generate_factor4lbkt, write2file, read_preprocessed_file
from lib.util.parse import get_statics4lbkt


def lbkt_general_config(local_params, global_params, global_objects):
    # 计算factor，如果已存在，直接读取
    dataset_name = local_params["dataset_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(
        global_params["datasets_config"]["train"]["setting_name"])
    train_file_name = global_params["datasets_config"]["train"]["file_name"]
    test_file_name = global_params["datasets_config"]["test"]["file_name"]
    statics4lbkt_path = os.path.join(setting_dir, train_file_name.replace(".txt", "_lbkt_statics.json"))

    if os.path.exists(statics4lbkt_path):
        statics4lbkt = load_json(statics4lbkt_path)
        use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict = {}, {}, {}, {}
        for k, v in statics4lbkt["use_time_mean_dict"].items():
            use_time_mean_dict[int(k)] = v
        for k, v in statics4lbkt["use_time_std_dict"].items():
            use_time_std_dict[int(k)] = v
        for k, v in statics4lbkt["num_attempt_mean_dict"].items():
            num_attempt_mean_dict[int(k)] = v
        for k, v in statics4lbkt["num_hint_mean_dict"].items():
            num_hint_mean_dict[int(k)] = v
    else:
        dataset_train = read_preprocessed_file(os.path.join(
            global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
            global_params["datasets_config"]["train"]["file_name"]
        ))
        dataset_train = dataset_delete_pad(dataset_train)
        use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict = \
            get_statics4lbkt(
                dataset_train, use_use_time_first=dataset_name in ["assist2009", "assist2012", "junyi2015"]
            )
        write_json({
            "use_time_mean_dict": use_time_mean_dict,
            "use_time_std_dict": use_time_std_dict,
            "num_attempt_mean_dict": num_attempt_mean_dict,
            "num_hint_mean_dict": num_hint_mean_dict
        }, statics4lbkt_path)

    train_dataset4lbkt_path = os.path.join(setting_dir, f"lbkt_dataset_{train_file_name}")
    if not os.path.exists(train_dataset4lbkt_path):
        dataset_train = read_preprocessed_file(os.path.join(
            global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
            global_params["datasets_config"]["train"]["file_name"]
        ))
        generate_factor4lbkt(
            dataset_train, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict,
            use_use_time_first=dataset_name in ["assist2009", "assist2012", "junyi2015"]
        )
        write2file(dataset_train, train_dataset4lbkt_path)

    test_dataset4lbkt_path = os.path.join(setting_dir, f"lbkt_dataset_{test_file_name}")
    if not os.path.exists(test_dataset4lbkt_path):
        dataset_test = read_preprocessed_file(os.path.join(
            global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
            global_params["datasets_config"]["test"]["file_name"]
        ))
        generate_factor4lbkt(
            dataset_test, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict,
            use_use_time_first=dataset_name in ["assist2009", "assist2012", "junyi2015"]
        )
        write2file(dataset_test, test_dataset4lbkt_path)

    if local_params["train_strategy"] == "valid_test":
        valid_file_name = global_params["datasets_config"]["valid"]["file_name"]
        valid_dataset4lbkt_path = os.path.join(setting_dir, f"lbkt_dataset_{valid_file_name}")
        if not os.path.exists(valid_dataset4lbkt_path):
            dataset_valid = read_preprocessed_file(os.path.join(
                global_objects["file_manager"].get_setting_dir(
                    global_params["datasets_config"]["train"]["setting_name"]),
                global_params["datasets_config"]["valid"]["file_name"]
            ))
            generate_factor4lbkt(
                dataset_valid, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict,
                use_use_time_first=dataset_name in ["assist2009", "assist2012", "junyi2015"]
            )
            write2file(dataset_valid, valid_dataset4lbkt_path)
        global_params["datasets_config"]["valid"]["file_name"] = os.path.basename(valid_dataset4lbkt_path)

    global_params["datasets_config"]["train"]["file_name"] = os.path.basename(train_dataset4lbkt_path)
    global_params["datasets_config"]["test"]["file_name"] = os.path.basename(test_dataset4lbkt_path)

    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "LBKT",
                "LBKT": {}
            }
        }
    }

    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_correct = local_params["num_correct"]
    dim_question = local_params["dim_question"]
    dim_correct = local_params["dim_correct"]
    dropout = local_params["dropout"]
    dim_h = local_params["dim_h"]
    dim_factor = local_params["dim_factor"]
    r = local_params["r"]
    d = local_params["d"]
    k = local_params["k"]
    b = local_params["b"]
    q_gamma = local_params["q_gamma"]

    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LBKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["num_correct"] = num_correct
    encoder_config["dim_question"] = dim_question
    encoder_config["dim_correct"] = dim_correct
    encoder_config["dropout"] = dropout
    encoder_config["dim_h"] = dim_h
    encoder_config["dim_factor"] = dim_factor
    encoder_config["r"] = r
    encoder_config["d"] = d
    encoder_config["k"] = k
    encoder_config["b"] = b
    encoder_config["q_gamma"] = q_gamma

    # q matrix
    global_objects["LBKT"] = {}
    global_objects["LBKT"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + q_gamma
    q_matrix = global_objects["LBKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    global_objects["logger"].info(
        f"model params\n    "
        f"num_concept: {num_concept}, num_question: {num_question}, num_correct: {num_correct}, \n    "
        f"dim_question: {dim_question}, dim_correct: {dim_correct}, dim_h: {dim_h}, dim_factor: {dim_factor}, "
        f"dropout: {dropout}, q_gamma: {q_gamma}, r: {r}, d: {d}, k:{k}, b:{b}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"LBKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def lbkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lbkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
