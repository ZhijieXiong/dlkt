import os.path

from ._config import *
from ._cl_config import *
from ._data_aug_config import *
from ._melt_config import *

from lib.template.params_template import PARAMS
from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS
from lib.template.model.DIMKT import MODEL_PARAMS as DIMKT_MODEL_PARAMS
from lib.util.basic import *
from lib.dataset.util import parse_difficulty
from lib.util.data import read_preprocessed_file, write_json, load_json


def dimkt_general_config(local_params, global_params, global_objects):
    global_params["models_config"]["kt_model"] = deepcopy(DIMKT_MODEL_PARAMS)
    global_params["models_config"]["kt_model"]["encoder_layer"]["type"] = "DIMKT"

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
    dimkt_encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
    dimkt_encoder_layer_config["num_concept"] = num_concept
    dimkt_encoder_layer_config["num_question"] = num_question
    dimkt_encoder_layer_config["num_question_diff"] = num_question_diff
    dimkt_encoder_layer_config["num_concept_diff"] = num_concept_diff
    dimkt_encoder_layer_config["dim_emb"] = dim_emb
    dimkt_encoder_layer_config["dropout"] = dropout

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
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@DIMKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def dimkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
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


def dimkt_instance_cl_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    instance_cl_general_config(local_params, global_params, global_objects)
    global_params["datasets_config"]["train"]["kt4aug"] = {"use_diff4dimkt": True}
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@DIMKT@@", "@@DIMKT-instance_cl@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def dimkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@DIMKT@@", "@@DIMKT-ME_adv_aug@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def dimkt_mutual_enhance4long_tail_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    mutual_enhance4long_tail_general_config(local_params, global_params, global_objects)

    # 需要在global_objects["mutual_enhance4long_tail"]["dataset_train"]里面添加上diff seq
    question_difficulty = global_objects["dimkt"]["question_difficulty"]
    concept_difficulty = global_objects["dimkt"]["concept_difficulty"]
    for item_data in global_objects["mutual_enhance4long_tail"]["dataset_train"]:
        item_data["question_diff_seq"] = []
        item_data["concept_diff_seq"] = []
        for q_id in item_data["question_seq"]:
            item_data["question_diff_seq"].append(question_difficulty[q_id])
        for c_id in item_data["concept_seq"]:
            item_data["concept_diff_seq"].append(concept_difficulty[c_id])
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_question_diff"] = max(
        question_difficulty.values()) + 1
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept_diff"] = max(
        concept_difficulty.values()) + 1

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@DIMKT@@", "@@DIMKT-ME4long_tail@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def dimkt_output_enhance_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    dimkt_general_config(local_params, global_params, global_objects)
    output_enhance_general_config(local_params, global_params, global_objects)

    global_params["datasets_config"]["train"]["kt_output_enhance"] = {"use_diff4dimkt": True}
    question_difficulty = global_objects["dimkt"]["question_difficulty"]
    concept_difficulty = global_objects["dimkt"]["concept_difficulty"]
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_question_diff"] = max(
        question_difficulty.values()) + 1
    global_params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["num_concept_diff"] = max(
        concept_difficulty.values()) + 1
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@DIMKT@@", "@@DIMKT-output_enhance@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
