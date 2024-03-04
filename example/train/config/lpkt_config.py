from ._config import *
from ._data_aug_config import *

from lib.template.objects_template import OBJECTS
from lib.template.params_template_v2 import PARAMS
from lib.util.basic import *
from lib.util.parse import cal_concept_difficulty
from lib.dataset.LPKTPlusUtil import LPKTPlusUtil
from lib.CONSTANT import INTERVAL_TIME4LPKT_PLUS, USE_TIME4LPKT_PLUS


def lpkt_general_config(local_params, global_params, global_objects):
    # 配置模型参数
    num_concept = local_params["num_concept"]
    num_question = local_params["num_question"]
    dim_e = local_params["dim_e"]
    dim_k = local_params["dim_k"]
    dim_correct = local_params["dim_correct"]
    dropout = local_params["dropout"]

    global_params["models_config"] = {
        "kt_model": {
            "type": "LPKT",
            "encoder_layer": {
                "LPKT": {}
            }
        }
    }
    encoder_layer_config = global_params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
    encoder_layer_config["num_concept"] = num_concept
    encoder_layer_config["num_question"] = num_question
    encoder_layer_config["dim_e"] = dim_e
    encoder_layer_config["dim_k"] = dim_k
    encoder_layer_config["dim_correct"] = dim_correct
    encoder_layer_config["dropout"] = dropout
    encoder_layer_config["ablation_set"] = local_params["ablation_set"]

    # q matrix
    global_objects["LPKT"] = {}
    global_objects["LPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + 0.03
    q_matrix = global_objects["LPKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    global_objects["logger"].info(
        "model params\n"
        f"    num of concept: {num_concept}, num of question: {num_question}, dim of e: {dim_e}, dim of k: {dim_k}, "
        f"dim of correct emb: {dim_correct}, dropout: {dropout}"
    )

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["save_model_dir_name"] = (
            f"{get_now_time().replace(' ', '@').replace(':', '-')}@@LPKT@@seed_{local_params['seed']}@@{setting_name}@@"
            f"{train_file_name.replace('.txt', '')}")


def lpkt_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        save_params(global_params, global_objects)

    return global_params, global_objects


def lpkt_max_entropy_adv_aug_config(local_params):
    global_params = deepcopy(PARAMS)
    global_objects = deepcopy(OBJECTS)
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    max_entropy_adv_aug_general_config(local_params, global_params, global_objects)
    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@LPKT@@", "@@LPKT-ME-ADA@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects


def lpkt_plus_config(local_params):
    global_params = {}
    global_objects = {}
    general_config(local_params, global_params, global_objects)
    lpkt_general_config(local_params, global_params, global_objects)
    global_objects["lpkt_plus_util"] = LPKTPlusUtil(global_params, global_objects)
    lpkt_plus_util = global_objects["lpkt_plus_util"]

    global_params["other"] = {"lpkt_plus": {}}
    global_params["datasets_config"]["train"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["train"]["lpkt_plus"] = {}
    global_params["datasets_config"]["test"]["type"] = "kt4lpkt_plus"
    global_params["datasets_config"]["test"]["kt4lpkt_plus"] = {}
    if local_params["train_strategy"] == "valid_test":
        global_params["datasets_config"]["valid"]["type"] = "kt4lpkt_plus"
        global_params["datasets_config"]["valid"]["kt4lpkt_plus"] = {}

    encoder_config = global_params["models_config"]["kt_model"]["encoder_layer"]
    global_params["models_config"]["kt_model"]["type"] = "LPKT_PLUS"
    encoder_config["LPKT_PLUS"] = deepcopy(encoder_config["LPKT"])
    del encoder_config["LPKT"]
    encoder_config["LPKT_PLUS"]["num_interval_time"] = len(INTERVAL_TIME4LPKT_PLUS)
    encoder_config["LPKT_PLUS"]["num_use_time"] = len(USE_TIME4LPKT_PLUS)
    encoder_config["LPKT_PLUS"]["ablation_set"] = local_params["ablation_set"]
    encoder_config["LPKT_PLUS"]["use_init_weight"] = local_params["use_init_weight"]

    global_objects["lpkt_plus"] = {}
    global_objects["lpkt_plus"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + local_params["gamma"]
    q_matrix = global_objects["lpkt_plus"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1
    del global_objects["LPKT"]

    # 统计习题难度
    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    global_objects["lpkt_plus"]["dataset_train"] = dataset_train
    if (local_params["w_que_diff_pred"] != 0) or (local_params["w_user_ability_pred"] != 0):
        global_params["other"]["lpkt_plus"]["min_fre4diff"] = local_params["min_fre4diff"]
        que_accuracy = lpkt_plus_util.cal_question_diff(dataset_train)
        que_difficulty = {k: 1 - v for k, v in que_accuracy.items()}
        global_objects["lpkt_plus"]["que_difficulty"] = que_difficulty
        if local_params["w_que_diff_pred"] != 0:
            global_objects["lpkt_plus"]["Q_table_mask"] = torch.from_numpy(
                global_objects["data"]["Q_table"]
            ).long().to(global_params["device"])
            global_objects["lpkt_plus"]["que_diff_ground_truth"] = torch.from_numpy(
                global_objects["data"]["Q_table"]
            ).float().to(global_params["device"])
            que_diff_ground_truth = global_objects["lpkt_plus"]["que_diff_ground_truth"]
            for q_id, que_diff in que_difficulty.items():
                que_diff_ground_truth[q_id] = que_diff_ground_truth[q_id] * que_diff
            global_objects["lpkt_plus"]["que_has_diff_ground_truth"] = torch.tensor(
                list(que_difficulty.keys())
            ).long().to(global_params["device"])

    # 统计习题区分度
    if local_params["w_que_disc_pred"] != 0:
        global_params["other"]["lpkt_plus"]["min_fre4disc"] = local_params["min_fre4disc"]
        global_params["other"]["lpkt_plus"]["min_seq_len4disc"] = local_params["min_seq_len4disc"]
        global_params["other"]["lpkt_plus"]["percent_threshold"] = local_params["percent_threshold"]
        que_discrimination = lpkt_plus_util.cal_que_discrimination(dataset_train, global_params["other"]["lpkt_plus"])
        que_has_disc_ground_truth = list(que_discrimination.keys())
        global_objects["lpkt_plus"]["que_has_disc_ground_truth"] = \
            torch.tensor(que_has_disc_ground_truth).long().to(global_params["device"])
        global_objects["lpkt_plus"]["que_disc_ground_truth"] = torch.tensor(
            [que_discrimination[q_id] * 10 for q_id in que_has_disc_ground_truth]
        ).float().to(global_params["device"])

    # 统计知识点正确率，用于初始化部分参数
    if local_params["use_init_weight"]:
        concept_accuracy = cal_concept_difficulty(dataset_train, {
          "data_type": global_params["datasets_config"]["data_type"],
          "num_concept": local_params["num_concept"],
          "num_min_concept": 50
        }, {"question2concept": global_objects["data"]["question2concept"]})
        lpkt_plus_util.get_user_proj_weight_init_value(concept_accuracy)

    # 提取学生认知标签和对应mask以及权重
    if local_params["w_user_ability_pred"] != 0:
        lpkt_plus_util.label_user_ability(dataset_train)

    # 损失权重配置
    global_params["loss_config"]["que diff pred loss"] = local_params["w_que_diff_pred"]
    global_params["loss_config"]["que disc pred loss"] = local_params["w_que_disc_pred"]
    global_params["loss_config"]["penalty neg loss"] = local_params["w_penalty_neg"]
    global_params["loss_config"]["learning loss"] = local_params["w_learning"]

    if local_params["save_model"]:
        global_params["save_model_dir_name"] = (
            global_params["save_model_dir_name"].replace("@@LPKT@@", "@@LPKT_PLUS@@"))
        save_params(global_params, global_objects)

    return global_params, global_objects
