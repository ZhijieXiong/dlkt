import json
import sys
import os
import inspect
import torch
import logging


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
sys.path.append(settings["LIB_PATH"])

from lib.util.FileManager import FileManager
from lib.util.basic import *
from lib.util.data import write_json
from lib.util.data import load_json
from lib.data_processor.load_raw import load_csv
from lib.util.parse import *
from lib.model.Module.KTEmbedLayer import KTEmbedLayer

from .util import config_optimizer


def general_config(local_params, global_params, global_objects):
    global_objects["logger"] = logging.getLogger("train_log")
    global_objects["logger"].setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(ch)
    file_manager = FileManager(FILE_MANAGER_ROOT)
    global_objects["file_manager"] = file_manager

    global_params["loss_config"] = {}
    global_params["other"] = {}
    global_params["save_model"] = local_params["save_model"]
    global_params["device"] = "cuda" if (
            torch.cuda.is_available() and not local_params.get("use_cpu", False)
    ) else "cpu"
    if local_params.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)
    global_params["seed"] = local_params["seed"]

    # 训练策略配置
    num_epoch = local_params["num_epoch"]
    use_early_stop = local_params["use_early_stop"]
    epoch_early_stop = local_params["epoch_early_stop"]
    use_last_average = local_params["use_last_average"]
    epoch_last_average = local_params["epoch_last_average"]
    main_metric = local_params["main_metric"]
    use_multi_metrics = local_params["use_multi_metrics"]
    mutil_metrics = local_params["multi_metrics"]
    train_strategy_type = local_params["train_strategy"]

    global_params["train_strategy"] = {}
    global_params["train_strategy"]["type"] = local_params["train_strategy"]
    train_strategy_config = global_params["train_strategy"]
    train_strategy_config["num_epoch"] = num_epoch
    train_strategy_config["main_metric"] = main_metric
    train_strategy_config["use_multi_metrics"] = use_multi_metrics
    if use_multi_metrics:
        train_strategy_config["multi_metrics"] = eval(mutil_metrics)
    else:
        train_strategy_config["multi_metrics"] = [main_metric]
    train_strategy_config["type"] = train_strategy_type
    train_strategy_config[train_strategy_type] = {}
    if train_strategy_type == "valid_test":
        train_strategy_config["valid_test"]["use_early_stop"] = use_early_stop
        if use_early_stop:
            train_strategy_config["valid_test"]["epoch_early_stop"] = epoch_early_stop
    elif train_strategy_type == "no_valid":
        train_strategy_config["no_valid"]["use_average"] = use_last_average
        if use_last_average:
            train_strategy_config["no_valid"]["epoch_last_average"] = epoch_last_average
    else:
        raise NotImplementedError()

    # 数据集配置
    setting_name = local_params["setting_name"]
    data_type = local_params["data_type"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    test_file_name = local_params["test_file_name"]

    global_params["datasets_config"] = {
        "train": {},
        "valid": {},
        "test": {},
        "data_type": local_params["data_type"]
    }
    datasets_config = global_params["datasets_config"]
    datasets_config["train"]["setting_name"] = setting_name
    datasets_config["test"]["setting_name"] = setting_name
    datasets_config["valid"]["setting_name"] = setting_name
    if train_strategy_config["type"] == "valid_test":
        datasets_config["valid"]["file_name"] = valid_file_name
    datasets_config["train"]["file_name"] = train_file_name
    datasets_config["test"]["file_name"] = test_file_name
    datasets_config["data_type"] = data_type

    # 是否使用样本加权
    use_sample_weight = local_params.get("use_sample_weight", False)
    sample_weight_method = local_params.get("sample_weight_method", "highlight_tail")
    tail_weight = local_params.get("tail_weight", 1)

    global_params["sample_weight"] = {
        "use_sample_weight": use_sample_weight,
        "sample_weight_method": sample_weight_method,
        "tail_weight": tail_weight
    }
    global_params["use_sample_weight"] = use_sample_weight
    global_params["sample_weight_method"] = sample_weight_method
    global_params["tail_weight"] = tail_weight

    # 是否用大模型的emb初始化
    use_LLM_emb4question = local_params.get("use_LLM_emb4question", False)
    use_LLM_emb4concept = local_params.get("use_LLM_emb4concept", False)
    train_LLM_emb = local_params.get("train_LLM_emb", False)

    global_params["LLM_emb_init"] = {
        "use_LLM_emb4question": use_LLM_emb4question,
        "use_LLM_emb4concept": use_LLM_emb4concept,
        "train_LLM_emb": train_LLM_emb
    }
    global_params["use_LLM_emb4question"] = use_LLM_emb4question
    global_params["use_LLM_emb4concept"] = use_LLM_emb4concept
    global_params["train_LLM_emb"] = train_LLM_emb

    global_objects["logger"].info(
        "basic setting\n"
        f"    device: {global_params['device']}, seed: {global_params['seed']}\n"
        "train policy\n"
        f"    type: {train_strategy_type}, {train_strategy_type}: {json.dumps(train_strategy_config[train_strategy_type])}, train batch size: {local_params['train_batch_size']}, num of epoch: {num_epoch}\n"
        f"    use sample weight: {use_sample_weight}{f', weight method: {sample_weight_method}' if use_sample_weight else ''}"
        f"{f', weight value for tail: {tail_weight}' if (use_sample_weight and sample_weight_method == 'highlight_tail') else ''}\n"
        f"embedding init\n"
        f"    use LLM emb to init question emb: {use_LLM_emb4question}, use LLM emb to init concept emb: {use_LLM_emb4concept}"
        f"{f', train LLM emb: {train_LLM_emb}' if (use_LLM_emb4question or use_LLM_emb4concept) else ''}\n"
        "evaluate metric\n"
        f"    main metric: {main_metric}, use multi metrics: {use_multi_metrics}{f', multi metrics: {mutil_metrics}' if use_multi_metrics else ''}"
    )

    # 优化器配置
    global_objects["logger"].info("optimizer setting")
    config_optimizer(local_params, global_params, global_objects, model_name="kt_model")

    # Q table，并解析Q table并得到相关数据
    dataset_name = local_params["dataset_name"]
    data_type = local_params["data_type"]
    Q_table = file_manager.get_q_table(dataset_name, data_type)
    global_objects["data"] = {}
    global_objects["data"]["Q_table"] = Q_table
    if Q_table is not None:
        global_objects["data"]["Q_table_tensor"] = \
            torch.from_numpy(global_objects["data"]["Q_table"]).long().to(global_params["device"])
        # 如果有Q table的话，可以分析出question2concept_list和concept2question_list
        # 前者index表示习题id，value表示该习题对应的知识点列表
        # 后者index表示知识点id，value表示该知识点对应的习题列表
        global_objects["data"]["question2concept"] = question2concept_from_Q(global_objects["data"]["Q_table"])
        global_objects["data"]["concept2question"] = concept2question_from_Q(global_objects["data"]["Q_table"])
        # 如果有Q table的话，q2c_table和q2c_mask_table都是(num_q, num_max_c)的tensor
        # num_max_c表示在该数据集中一道习题最多对应几个知识点
        q2c_table, q2c_mask_table, num_max_concept = KTEmbedLayer.parse_Q_table(Q_table, global_params["device"])
        global_objects["data"]["q2c_table"] = q2c_table
        global_objects["data"]["q2c_mask_table"] = q2c_mask_table
        global_objects["data"]["num_max_concept"] = num_max_concept

    # 获取习题和知识点组合的关系
    Q_table_single_concept = file_manager.get_q_table(dataset_name, "single_concept")
    if Q_table_single_concept is not None:
        question2concept_combination = question2concept_from_Q(Q_table_single_concept)
        global_objects["data"]["question2concept_combination"] = question2concept_combination

    global_objects["logger"].info(
        "dataset\n"
        f"    setting: {setting_name}, dataset: {dataset_name}, data type: {data_type}, "
        f"train: {train_file_name}{f', valid: {valid_file_name}' if train_strategy_type == 'valid_test' else ''}, test: {test_file_name}"
    )

    # 数据集统计信息
    statics_info_file_path = os.path.join(
        file_manager.get_setting_dir(setting_name),
        datasets_config["train"]["file_name"].replace(".txt", f"_statics_special.json")
    )
    if not os.path.exists(statics_info_file_path):
        global_objects["logger"].warning(
            f"\nWARNING: statics of train dataset (`{statics_info_file_path}`) is not exist. This file is required for some cases, e.g., "
            "fine grain evaluation such as long tail problem and some model using transfer_head2zero. "
            "If it is necessary, please run `prepare4fine_trained_evaluate.py` to generate statics of train dataset\n"
        )
    else:
        with open(statics_info_file_path, "r") as file:
            global_objects["data"]["train_data_statics"] = json.load(file)

    # 训练集习题正确率信息
    statics_statics_common_path = os.path.join(
        file_manager.get_setting_dir(setting_name),
        datasets_config["train"]["file_name"].replace(".txt", f"_statics_common.json")
    )
    if not os.path.exists(statics_statics_common_path):
        global_objects["logger"].warning(
            f"\nWARNING: statics of train dataset (`{statics_statics_common_path}`) is not exist. This file is required "
            f"for IPS-question"
            "If it is necessary, please run `prepare4fine_trained_evaluate.py` to generate statics of train dataset\n"
        )
    else:
        with open(statics_statics_common_path, "r") as file:
            train_statics_common = json.load(file)
            question_acc_dict = {}
            for q_id, q_acc in train_statics_common["question_acc"].items():
                question_acc_dict[int(q_id)] = q_acc
            train_statics_common["question_acc"] = question_acc_dict
            global_objects["data"]["train_data_statics_common"] = train_statics_common

    # load大模型的embedding
    if use_LLM_emb4question or use_LLM_emb4concept:
        dataset_name = local_params["dataset_name"]
        data_type = local_params["data_type"]
        data_dir = global_objects["file_manager"].get_preprocessed_dir(dataset_name)
        if use_LLM_emb4concept:
            global_objects["data"]["LLM_concept_embeddings"] = (
                load_json(os.path.join(data_dir, "concept_embeddings.json")))
        if use_LLM_emb4question:
            # 目前没有习题的信息，先用知识点顶着
            global_objects["data"]["LLM_question_embeddings"] = (
                load_json(os.path.join(data_dir, "concept_embeddings.json")))
        if data_type == "only_question":
            concept_id_map = load_csv(os.path.join(data_dir, "concept_id_map_multi_concept.csv"))
            # todo
        else:
            # 单知识点数据集
            concept_id_map = load_csv(os.path.join(data_dir, "concept_id_map_single_concept.csv"))
            if use_LLM_emb4question:
                q_id2original_c_id = []
                for q_id in range(len(global_objects["data"]["Q_table"])):
                    c_id = get_concept_from_question(q_id, global_objects["data"]["Q_table"])[0]
                    original_c_id = concept_id_map[concept_id_map["concept_mapped_id"] == c_id]["concept_id"].iloc[0]
                    q_id2original_c_id.append(original_c_id)
                global_objects["data"]["q_id2original_c_id"] = q_id2original_c_id

            if use_LLM_emb4concept:
                c_id2original_c_id = {}
                for _, row in concept_id_map.iterrows():
                    c_id = row["concept_mapped_id"]
                    original_c_id = row["concept_id"]
                    c_id2original_c_id[c_id] = original_c_id
                global_objects["data"]["c_id2original_c_id"] = c_id2original_c_id


def save_params(global_params, global_objects):
    file_manager = global_objects["file_manager"]
    model_root_dir = file_manager.get_models_dir()
    model_dir_name = global_params["save_model_dir_name"]
    model_dir = os.path.join(model_root_dir, model_dir_name)
    global_params["save_model_dir"] = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        assert False, f"{model_dir} exists"

    params_path = os.path.join(model_dir, "params.json")
    params_json = params2str(global_params)
    write_json(params_json, params_path)

    log_path = os.path.join(model_dir, "train_log.txt")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    global_objects["logger"].addHandler(fh)
