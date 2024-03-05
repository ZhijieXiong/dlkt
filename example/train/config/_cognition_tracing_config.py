import torch
import os

from lib.util.data import read_preprocessed_file
from lib.dataset.CognitionTracingUtil import CognitionTracingUtil


def cognition_tracing_general_config(local_params, global_params, global_objects):
    global_objects["cognition_tracing_util"] = CognitionTracingUtil(global_params, global_objects)
    cognition_tracing_util = global_objects["cognition_tracing_util"]

    global_objects["cognition_tracing"] = {}
    global_objects["cognition_tracing"]["q_matrix"] = torch.from_numpy(
        global_objects["data"]["Q_table"]
    ).float().to(global_params["device"]) + local_params["gamma"]
    q_matrix = global_objects["cognition_tracing"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    # 统计习题难度
    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    global_objects["cognition_tracing"]["dataset_train"] = dataset_train
    if (local_params["w_que_diff_pred"] != 0) or (local_params["w_user_ability_pred"] != 0):
        global_params["other"]["cognition_tracing"] = {}
        global_params["other"]["cognition_tracing"]["min_fre4diff"] = local_params["min_fre4diff"]
        que_accuracy = cognition_tracing_util.cal_question_diff(dataset_train)
        que_difficulty = {k: 1 - v for k, v in que_accuracy.items()}
        global_objects["cognition_tracing"]["que_difficulty"] = que_difficulty
        if local_params["w_que_diff_pred"] != 0:
            global_objects["cognition_tracing"]["Q_table_mask"] = torch.from_numpy(
                global_objects["data"]["Q_table"]
            ).long().to(global_params["device"])
            global_objects["cognition_tracing"]["que_diff_ground_truth"] = torch.from_numpy(
                global_objects["data"]["Q_table"]
            ).float().to(global_params["device"])
            que_diff_ground_truth = global_objects["cognition_tracing"]["que_diff_ground_truth"]
            for q_id, que_diff in que_difficulty.items():
                que_diff_ground_truth[q_id] = que_diff_ground_truth[q_id] * que_diff
            global_objects["cognition_tracing"]["que_has_diff_ground_truth"] = torch.tensor(
                list(que_difficulty.keys())
            ).long().to(global_params["device"])

    # 统计习题区分度
    if local_params["w_que_disc_pred"] != 0:
        global_params["other"]["cognition_tracing"] = {}
        global_params["other"]["cognition_tracing"]["min_fre4disc"] = local_params["min_fre4disc"]
        global_params["other"]["cognition_tracing"]["min_seq_len4disc"] = local_params["min_seq_len4disc"]
        global_params["other"]["cognition_tracing"]["percent_threshold"] = local_params["percent_threshold"]
        que_discrimination = \
            cognition_tracing_util.cal_que_discrimination(dataset_train, global_params["other"]["cognition_tracing"])
        que_has_disc_ground_truth = list(que_discrimination.keys())
        global_objects["cognition_tracing"]["que_has_disc_ground_truth"] = \
            torch.tensor(que_has_disc_ground_truth).long().to(global_params["device"])
        global_objects["cognition_tracing"]["que_disc_ground_truth"] = torch.tensor(
            [que_discrimination[q_id] * 10 for q_id in que_has_disc_ground_truth]
        ).float().to(global_params["device"])

    # # 统计知识点正确率，用于初始化部分参数
    # if local_params["use_init_weight"]:
    #     concept_accuracy = cal_concept_difficulty(dataset_train, {
    #       "data_type": global_params["datasets_config"]["data_type"],
    #       "num_concept": local_params["num_concept"],
    #       "num_min_concept": 50
    #     }, {"question2concept": global_objects["data"]["question2concept"]})
    #     cognition_tracing_util.get_user_proj_weight_init_value(concept_accuracy)
    #
    # # 提取学生认知标签和对应mask以及权重
    # if local_params["w_user_ability_pred"] != 0:
    #     cognition_tracing_util.label_user_ability(dataset_train)

    # 损失权重配置
    global_params["loss_config"]["que diff pred loss"] = local_params["w_que_diff_pred"]
    global_params["loss_config"]["que disc pred loss"] = local_params["w_que_disc_pred"]
    global_params["loss_config"]["penalty neg loss"] = local_params["w_penalty_neg"]
    global_params["loss_config"]["learning loss"] = local_params["w_learning"]
