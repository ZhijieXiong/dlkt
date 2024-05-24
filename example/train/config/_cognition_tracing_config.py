import torch
import os
import numpy as np

from lib.util.data import read_preprocessed_file
from lib.util.parse import cal_concept_acc
from lib.dataset.CognitionTracingUtil import CognitionTracingUtil


def cognition_tracing_general_config(local_params, global_params, global_objects):
    global_objects["cognition_tracing_util"] = CognitionTracingUtil(global_params, global_objects)
    cognition_tracing_util = global_objects["cognition_tracing_util"]

    global_objects["cognition_tracing"] = {}
    global_params["other"]["cognition_tracing"] = {}

    dataset_train = read_preprocessed_file(os.path.join(
        global_objects["file_manager"].get_setting_dir(global_params["datasets_config"]["train"]["setting_name"]),
        global_params["datasets_config"]["train"]["file_name"]
    ))
    global_objects["cognition_tracing"]["dataset_train"] = dataset_train

    w_que_diff_pred = local_params.get("w_que_diff_pred", 0)
    w_que_disc_pred = local_params.get("w_que_disc_pred", 0)
    w_user_ability_pred = local_params.get("w_user_ability_pred", 0)

    # 统计习题难度
    if (w_que_diff_pred != 0) or (w_user_ability_pred != 0):
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
    if w_que_disc_pred != 0:
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

    # 配置多知识点习题penalty损失权重
    Q_table = global_objects["data"]["Q_table"]
    qc_counts = Q_table.sum(axis=-1)
    # 原始方法
    penalty_weight4question = torch.from_numpy(np.exp(1 - qc_counts)).to(global_params["device"])
    # 考虑到一道题对应知识点越多，习题越难，那么可能就越要求每个知识点掌握程度都高，这个权重应该是先下降再上升
    global_objects["data"]["penalty_weight4question"] = penalty_weight4question
    mask4single_concept = torch.from_numpy(qc_counts <= 1).to(global_params["device"])
    global_objects["data"]["mask4single_concept"] = mask4single_concept

    # 提取学生认知标签和对应mask以及权重
    if w_user_ability_pred != 0:
        cognition_tracing_util.label_user_ability(dataset_train)

    # 单阶段还是多阶段；损失权重配置
    multi_stage = local_params.get("multi_stage", False)
    use_hard_Q_table = local_params.get("use_hard_Q_table", True)
    q_table_loss_th = local_params.get("q_table_loss_th", 0.5)
    use_pretrain = local_params.get("use_pretrain", False)
    epoch_pretrain = local_params.get("epoch_pretrain", 20)
    w_learning = local_params.get("w_learning", 0)
    w_counter_fact = local_params.get("w_counter_fact", 0)
    w_penalty_neg = local_params.get("w_penalty_neg", 0)
    w_q_table = local_params.get("w_q_table", 0)
    w_unbiased_cl = local_params.get("w_unbiased_cl", 0)
    temp = local_params.get("temp", 0.05)
    correct_noise_strength = local_params.get("correct_noise_strength", 0.1)

    # 统计知识点难度，用于初始化encoder
    if use_pretrain:
        concept_acc_dict = cal_concept_acc(dataset_train, {
            "num_min_concept": 50,
            "num_concept": local_params["num_concept"],
            "data_type": local_params["data_type"]
        }, global_objects["data"])
        user_ability_init = [0.] * local_params["num_concept"]
        for c_id, c_acc in concept_acc_dict.items():
            if c_acc >= 0:
                user_ability_init[c_id] = c_acc / 4
            else:
                user_ability_init[c_id] = 0.05
        global_objects["cognition_tracing"]["user_ability_init"] = \
            torch.tensor(user_ability_init).float().to(global_params["device"])

    global_params["other"]["cognition_tracing"]["multi_stage"] = multi_stage
    global_params["other"]["cognition_tracing"]["use_pretrain"] = use_pretrain
    global_params["other"]["cognition_tracing"]["epoch_pretrain"] = epoch_pretrain
    global_params["other"]["cognition_tracing"]["use_hard_Q_table"] = use_hard_Q_table
    global_params["other"]["cognition_tracing"]["q_table_loss_th"] = q_table_loss_th
    if w_que_diff_pred != 0:
        global_params["loss_config"]["que diff pred loss"] = w_que_diff_pred
    if w_que_disc_pred != 0:
        global_params["loss_config"]["que disc pred loss"] = w_que_disc_pred
    if w_user_ability_pred != 0:
        global_params["loss_config"]["user ability pred loss"] = w_user_ability_pred
    if w_q_table != 0:
        global_params["loss_config"]["q table loss"] = w_q_table
    if w_penalty_neg != 0:
        global_params["loss_config"]["penalty neg loss"] = w_penalty_neg
    if w_learning != 0:
        global_params["loss_config"]["learning loss"] = w_learning
    if w_counter_fact != 0:
        global_params["loss_config"]["counterfactual loss"] = w_counter_fact
    if w_unbiased_cl != 0:
        global_params["loss_config"]["unbiased cl loss"] = w_unbiased_cl
    global_params["other"]["instance_cl"] = {
        "temp": temp,
        "correct_noise_strength": correct_noise_strength
    }

    global_objects["logger"].info(
        "loss weight params\n"
        f"    w_unbiased_cl: {w_unbiased_cl}, w_que_diff_pred: {w_que_diff_pred}, w_que_disc_pred: {w_que_disc_pred}, "
        f"w_user_ability_pred: {w_user_ability_pred}, w_penalty_neg: {w_penalty_neg}, w_learning: {w_learning}, "
        f"w_counter_fact: {w_counter_fact}, w_q_table: {w_q_table}\n"
        f"other params:\n"
        f"    multi_stage: {multi_stage}, use_hard_Q_table: {use_hard_Q_table}, use_pretrain: {use_pretrain}, "
        f"epoch_pretrain: {epoch_pretrain}, q_table_loss_th: {q_table_loss_th}, temp: {temp}, "
        f"correct_noise_strength: {correct_noise_strength}"
    )
