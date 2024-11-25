import os.path

from ._config import *

from lib.util.basic import *
from lib.dataset.util import parse_difficulty
from lib.util.data import read_preprocessed_file, write_json, load_json


def dimkt_que_general_config(local_params, global_params, global_objects):
    global_params["models_config"] = {
        "kt_model": {
            "kt_embed_layer": {},
            "encoder_layer": {
                "type": "DIMKT",
                "DIMKT": {}
            }
        }
    }

    # 配置模型参数和数据集参数
    dataset_name = local_params["dataset_name"]
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    use_concept = local_params["use_concept"]
    que_emb_file_name = local_params["que_emb_file_name"]
    frozen_que_emb = local_params["frozen_que_emb"]
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

    # embed layer
    que_emb_path = os.path.join(
        global_objects["file_manager"].get_preprocessed_dir(dataset_name), que_emb_file_name
    )
    kt_embed_layer_config = global_params["models_config"]["kt_model"]["kt_embed_layer"]
    kt_embed_layer_config["question"] = {
        "num_item": num_question,
        "dim_item": dim_emb,
        "learnable": not frozen_que_emb,
        "embed_path": que_emb_path
    }
    kt_embed_layer_config["question_difficulty"] = {
        "num_item": max(question_difficulty.values()) + 1,
        "dim_item": dim_emb
    }
    kt_embed_layer_config["concept"] = {
        "num_item": num_concept,
        "dim_item": dim_emb,
    }
    kt_embed_layer_config["concept_difficulty"] = {
        "num_item": max(concept_difficulty.values()) + 1,
        "dim_item": dim_emb,
    }
    if not use_concept:
        kt_embed_layer_config["concept"]["init_method"] = "zero"
        kt_embed_layer_config["concept"]["learnable"] = False
        kt_embed_layer_config["concept_difficulty"]["init_method"] = "zero"
        kt_embed_layer_config["concept_difficulty"]["learnable"] = False
    kt_embed_layer_config["correct"] = {
        "num_item": 2,
        "dim_item": dim_emb,
    }

    # encoder layer
    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
    encoder_config["num_concept"] = num_concept
    encoder_config["num_question"] = num_question
    encoder_config["num_question_diff"] = num_question_diff
    encoder_config["num_concept_diff"] = num_concept_diff
    encoder_config["dim_emb"] = dim_emb
    encoder_config["dropout"] = dropout

    global_objects["logger"].info(
        f"model params\n    "
        f"num of concept: {num_concept}, num of question: {num_question}, frozen_que_emb: {frozen_que_emb}, "
        f"que_emb_path: {que_emb_path}\n    "
        f"num of min question: {num_min_question}, num of min concept: {num_min_concept}, "
        f"num of question difficulty: {num_question_diff}, num of concept difficulty: {num_concept_diff}, "
        f"dim of emb: {dim_emb}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"DIMKT_QUE@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")


def dimkt_que_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    dimkt_que_general_config(local_params, global_params, global_objects)

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

    # # 需要改一下DIMKT的模型参数
    # question_difficulty = global_objects["dimkt"]["question_difficulty"]
    # concept_difficulty = global_objects["dimkt"]["concept_difficulty"]
    # global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_question_diff"] = max(
    #     question_difficulty.values()) + 1
    # global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept_diff"] = max(
    #     concept_difficulty.values()) + 1

    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects
