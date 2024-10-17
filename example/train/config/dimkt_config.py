import os.path

from ._config import *

from lib.util.basic import *
from lib.dataset.util import parse_difficulty
from lib.util.data import read_preprocessed_file, write_json, load_json


def dimkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "encoder_layer": {
                "type": "DIMKT",
                "DIMKT": {}
            }
        }
    }

    # 配置模型参数和数据集参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    num_min_question = local_params["num_min_question"]
    num_min_concept = local_params["num_min_concept"]
    num_question_diff = local_params["num_question_diff"]
    num_concept_diff = local_params["num_concept_diff"]
    dim_emb = local_params["dim_emb"]
    dropout = local_params["dropout"]

    # 统计习题和知识点难度信息，如果已存在，直接读取
    setting_dir = global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"])
    train_file_name = global_params["datasets_config"]["train"]["file_name"]
    difficulty_info_path = os.path.join(setting_dir, train_file_name.replace(".txt", "_dimkt_diff.json"))
    if os.path.exists(difficulty_info_path):
        difficulty_info = load_json(difficulty_info_path)
        question_difficulty = {}
        concept_difficulty = {}
        for k, v in difficulty_info["question_difficulty"].items():
            question_difficulty[int(k)] = v
        for k, v in difficulty_info["concept_difficulty"].items():
            concept_difficulty[int(k)] = v
    else:
        dataset_train = read_preprocessed_file(os.path.join(
            global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
            global_params["datasets_config"]["train"]["file_name"]
        ))
        parse_difficulty_params = {
            "data_type": global_params["datasets_config"]["data_type"],
            "num_min_question": num_min_question,
            "num_min_concept": num_min_concept,
            "num_question_diff": num_question_diff,
            "num_concept_diff": num_concept_diff,
            "num_concept": num_concept,
            "num_question": num_question
        }
        question_difficulty, concept_difficulty = \
            parse_difficulty(dataset_train, parse_difficulty_params, global_objects["data"])
        difficulty_info = {"question_difficulty": question_difficulty, "concept_difficulty": concept_difficulty}
        write_json(difficulty_info, difficulty_info_path)

    global_objects["dimkt"] = {}
    global_objects["dimkt"]["question_difficulty"] = question_difficulty
    global_objects["dimkt"]["concept_difficulty"] = concept_difficulty
    diff_fuse_table = [0] * num_concept
    for c_id, c_diff_id in concept_difficulty.items():
        diff_fuse_table[c_id] = c_diff_id
    global_objects["dimkt"]["diff_fuse_table"] = torch.LongTensor(diff_fuse_table).to(global_params["device"])
    global_params["datasets_config"]["train"]["type"] = "kt4dimkt"
    global_params["datasets_config"]["train"]["kt4dimkt"] = {}
    global_params["datasets_config"]["train"]["kt4dimkt"]["num_question_difficulty"] = num_question_diff
    global_params["datasets_config"]["train"]["kt4dimkt"]["num_concept_difficulty"] = num_concept_diff
    global_params["datasets_config"]["test"]["type"] = "kt4dimkt"
    global_params["datasets_config"]["test"]["kt4dimkt"] = {}
    global_params["datasets_config"]["test"]["kt4dimkt"]["num_question_difficulty"] = num_question_diff
    global_params["datasets_config"]["test"]["kt4dimkt"]["num_concept_difficulty"] = num_concept_diff
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "kt4dimkt"
        global_params["datasets_config"]["valid"]["kt4dimkt"] = {}
        global_params["datasets_config"]["valid"]["kt4dimkt"]["num_question_difficulty"] = num_question_diff
        global_params["datasets_config"]["valid"]["kt4dimkt"]["num_concept_difficulty"] = num_concept_diff

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["num_question_diff"] = num_question_diff
    encoder_config["num_concept_diff"] = num_concept_diff
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, num of min question: {num_min_question}, "
        f"num of min concept: {num_min_concept}, num of question difficulty: {num_question_diff}, "
        f"num of concept difficulty: {num_concept_diff}, dim of emb: {dim_emb}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DIMKT@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dimkt_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)

    # sample_reweight
    sample_reweight = {
        "use_sample_reweight": local_params["use_sample_reweight"],
        "sample_reweight_method": local_params["sample_reweight_method"],
    }
    use_sample_reweight = local_params["use_sample_reweight"]
    sample_reweight_method = local_params["sample_reweight_method"]
    use_IPS = False
    if use_sample_reweight and local_params["sample_reweight_method"] in ["IPS-double", "IPS-seq", "IPS-question"]:
        use_IPS = True
        sample_reweight["IPS_min"] = local_params["IPS_min"]
        sample_reweight["IPS_his_seq_len"] = local_params['IPS_his_seq_len']
    global_params["sample_reweight"] = sample_reweight

    global_objects["logger"].info(
        f"sample weight\n    "
        f"use_sample_reweight: {use_sample_reweight}, sample_reweight_method: {sample_reweight_method}"
        f"{', IPS_min: ' + str(local_params['IPS_min']) + ', IPS_his_seq_len: ' + str(local_params['IPS_his_seq_len']) if use_IPS else ''}"
    )

    # 需要改一下DIMKT的模型参数
    question_difficulty = global_objects["dimkt"]["question_difficulty"]
    concept_difficulty = global_objects["dimkt"]["concept_difficulty"]
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_question_diff"] = max(
        question_difficulty.values()) + 1
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept_diff"] = max(
        concept_difficulty.values()) + 1

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def dimkt_core_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    # 需要改一下DIMKT的模型参数
    question_difficulty = global_objects["dimkt"]["question_difficulty"]
    concept_difficulty = global_objects["dimkt"]["concept_difficulty"]
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_question_diff"] = max(
        question_difficulty.values()) + 1
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept_diff"] = max(
        concept_difficulty.values()) + 1

    fusion_mode = local_params["fusion_mode"]
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["fusion_mode"] = fusion_mode
    global_objects["logger"].info(
        "core params\n"
        f"    fusion_mode: {fusion_mode}"
    )

    global_params["loss_config"]["KL loss"] = 1

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("DIMKT@@", "DIMKT-CORE@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
