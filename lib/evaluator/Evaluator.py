import json
import os.path

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from .util import *
from ..util.basic import get_now_time
from ..model.util import get_mask4last_or_penultimate


def get_performance(item_list, item_pg_all):
    """
    获取指定item（习题或者知识点）上的性能
    :param item_list: 习题或者知识点的id list，即想要获取的那部分
    :param item_pg_all: 形式为{item1: [(p1, g1), (p2, g2), ...], item2: [(p1, g1), (p2, g2), ...], ...}
    :return:
    """
    predict_score = []
    ground_truth = []

    for item in item_list:
        p_list = list(map(lambda x: x[0], item_pg_all[item]))
        g_list = list(map(lambda x: x[1], item_pg_all[item]))
        predict_score += p_list
        ground_truth += g_list

    if len(ground_truth) == 0:
        return {
            "num_sample": 0,
            "AUC": 0.,
            "ACC": 0.,
            "RMSE": 0.,
            "MAE": 0.
        }

    AUC = roc_auc_score(y_true=ground_truth, y_score=predict_score)
    predict_label = [1 if p >= 0.5 else 0 for p in predict_score]
    ACC = accuracy_score(y_true=ground_truth, y_pred=predict_label)
    MAE = mean_absolute_error(y_true=ground_truth, y_pred=predict_score)
    RMSE = mean_squared_error(y_true=ground_truth, y_pred=predict_score) ** 0.5

    return {
        "num_sample": len(predict_score),
        "AUC": AUC,
        "ACC": ACC,
        "RMSE": RMSE,
        "MAE": MAE
    }


def get_performance_qc(question_list, concept_list, qc_pg_all):
    """
    获取指定习题以及知识点上的性能
    :param question_list:
    :param concept_list:
    :param qc_pg_all: 形式为{"q1_c1": [(p1, g1), (p2, g2), ...], "q2_c2": [(p1, g1), (p2, g2), ...], ...}
    :return:
    """
    predict_score = []
    ground_truth = []

    qc_all = []
    for q_id in question_list:
        for c_id in concept_list:
            qc_all.append(f"{q_id}_{c_id}")

    for qc_id in qc_all:
        p_list = list(map(lambda x: x[0], qc_pg_all[qc_id]))
        g_list = list(map(lambda x: x[1], qc_pg_all[qc_id]))
        predict_score += p_list
        ground_truth += g_list

    AUC = roc_auc_score(y_true=ground_truth, y_score=predict_score)
    predict_label = [1 if p >= 0.5 else 0 for p in predict_score]
    ACC = accuracy_score(y_true=ground_truth, y_pred=predict_label)
    MAE = mean_absolute_error(y_true=ground_truth, y_pred=predict_score)
    RMSE = mean_squared_error(y_true=ground_truth, y_pred=predict_score) ** 0.5

    return {
        "num_sample": len(predict_score),
        "AUC": AUC,
        "ACC": ACC,
        "RMSE": RMSE,
        "MAE": MAE
    }


class Evaluator:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    def evaluate(self):
        use_transfer = self.params["transfer_head2zero"]["use_transfer"]
        fine_grain_config = self.params["evaluate"]["fine_grain"]
        max_seq_len = fine_grain_config["max_seq_len"]
        seq_len_absolute = fine_grain_config["seq_len_absolute"]
        data_type = self.params["datasets_config"]["data_type"]
        all_label_dis, all_score_dis = [[] for _ in range(max_seq_len - 1)], [[] for _ in range(max_seq_len - 1)]

        model = self.objects["models"]["kt_model"]
        data_loader = self.objects["data_loaders"]["test_loader"]
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            question_all = []
            concept_all = []
            result_all_batch = []
            if use_transfer and hasattr(model, "set_emb4zero"):
                model.set_emb4zero()
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                question_seq = batch["question_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)

                predict_score_seq_len_minus1 = model.get_predict_score_seq_len_minus1(batch)
                result_all_batch.append({
                    "label": correct_seq[:, 1:].detach().cpu().numpy().tolist(),
                    "predict_score": predict_score_seq_len_minus1.detach().cpu().numpy().tolist(),
                    "correct_seq": batch["correct_seq"].detach().cpu().numpy().tolist(),
                    "mask_seq": batch["mask_seq"].detach().cpu().numpy().tolist()
                })
                label_dis, score_dis = record_dis4seq_len(correct_seq[:, 1:],
                                                          predict_score_seq_len_minus1,
                                                          batch["mask_seq"][:, 1:])
                for i in range(max_seq_len-1):
                    all_score_dis[i] += score_dis[i]
                    all_label_dis[i] += label_dis[i]

                if use_transfer and hasattr(model, "get_predict_score4question_zero"):
                    predict_score = model.get_predict_score4question_zero(batch).detach().cpu().numpy()
                else:
                    predict_score = model.get_predict_score(batch).detach().cpu().numpy()
                ground_truth = torch.masked_select(correct_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)

                question_all.append(torch.masked_select(question_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy())
                if data_type != "only_question":
                    concept_all.append(torch.masked_select(batch["concept_seq"][:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy())

            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5

        # overall performance
        self.objects["logger"].info(f"overall performance\nAUC: {AUC:<9.5}, ACC: {ACC:<9.5}, RMSE: {MAE:<9.5}, MAE: {RMSE:<9.5}\n")

        # performance by seq len
        label_dis4len, score_dis4len, indices4len = evaluate4seq_len(all_label_dis, all_score_dis, seq_len_absolute)
        self.objects["logger"].info("split by seq length")
        for i in range(len(label_dis4len)):
            if len(label_dis4len[i]) == 0:
                continue
            g = np.array(label_dis4len[i])
            p = np.array(score_dis4len[i])
            p_label = [1 if _ >= 0.5 else 0 for _ in p]
            answer_acc = g.sum() / len(g)
            self.objects["logger"].info(
                f"({indices4len[i][0]:<3}, {indices4len[i][1]:<3}), num of samples is {g.size:<10}, "
                f"acc of answer is {answer_acc * 100:<4.3}%: "
                f"auc is {roc_auc_score(y_true=g, y_score=p):<9.5}, "
                f"acc is {accuracy_score(g, p_label):<9.5}"
            )

        # evaluate_bias(result_all_batch, 3)

        statics_path = fine_grain_config["statics_path"]
        if not os.path.exists(statics_path):
            return

        result4statics = {}
        with open(statics_path, "r") as f:
            statics_train = json.load(f)

        question_all = np.concatenate(question_all, axis=0)
        all_question_dis = defaultdict(list)
        for q_id, p, g in zip(question_all, predict_score_all, ground_truth_all):
            all_question_dis[q_id].append((p, g))

        if model.objects["data"].get("question_no_head_qs", False):
            # performance of zero shot questions of not-head questions sharing one concept
            performance4zero_q = get_performance(model.objects["data"].get("question_no_head_qs"), all_question_dis)
            self.objects["logger"].info(
                f"\nzero shot questions of no head questions sharing one concept\n"
                f"zero shot ({performance4zero_q['num_sample']:<9} samples), "
                f"AUC: {performance4zero_q['AUC']:<9.6}, "
                f"ACC: {performance4zero_q['ACC']:<9.6}, "
                f"RMSE: {performance4zero_q['RMSE']:<9.6}, "
                f"MAE: {performance4zero_q['MAE']:<9.6}\n"
            )
        else:
            self.objects["logger"].info("")

        result4statics['question_zero_fre'] = get_performance(statics_train['question_zero_fre'], all_question_dis)
        result4statics['question_low_fre'] = get_performance(statics_train['question_low_fre'], all_question_dis)
        result4statics['question_middle_fre'] = get_performance(statics_train['question_middle_fre'], all_question_dis)
        result4statics['question_high_fre'] = get_performance(statics_train['question_high_fre'], all_question_dis)
        result4statics['question_low_acc'] = get_performance(statics_train['question_low_acc'], all_question_dis)
        result4statics['question_middle_acc'] = get_performance(statics_train['question_middle_acc'], all_question_dis)
        result4statics['question_high_acc'] = get_performance(statics_train['question_high_acc'], all_question_dis)

        self.objects["logger"].info(
            f"question frequency\n"
            f"zero shot ({result4statics['question_zero_fre']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_zero_fre']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_zero_fre']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_zero_fre']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_zero_fre']['MAE']:<9.6}\n"
            f"low frequency ({result4statics['question_low_fre']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_low_fre']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_low_fre']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_low_fre']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_low_fre']['MAE']:<9.6}\n"
            f"middle frequency ({result4statics['question_middle_fre']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_middle_fre']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_middle_fre']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_middle_fre']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_middle_fre']['MAE']:<9.6}\n"
            f"high frequency ({result4statics['question_high_fre']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_high_fre']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_high_fre']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_high_fre']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_high_fre']['MAE']:<9.6}\n\n"
            f"question accuracy\n"
            f"low accuracy ({result4statics['question_low_acc']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_low_acc']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_low_acc']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_low_acc']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_low_acc']['MAE']:<9.6}\n"
            f"middle accuracy ({result4statics['question_middle_acc']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_middle_acc']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_middle_acc']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_middle_acc']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_middle_acc']['MAE']:<9.6}\n"
            f"high accuracy ({result4statics['question_high_acc']['num_sample']:<9} samples), "
            f"AUC: {result4statics['question_high_acc']['AUC']:<9.6}, "
            f"ACC: {result4statics['question_high_acc']['ACC']:<9.6}, "
            f"RMSE: {result4statics['question_high_acc']['RMSE']:<9.6}, "
            f"MAE: {result4statics['question_high_acc']['MAE']:<9.6}"
        )

        if data_type != "only_question":
            concept_all = np.concatenate(concept_all, axis=0)
            all_concept_dis = defaultdict(list)
            for c_id, p, g in zip(concept_all, predict_score_all, ground_truth_all):
                all_concept_dis[c_id].append((p, g))
            result4statics['concept_low_fre'] = get_performance(statics_train['concept_low_fre'], all_concept_dis)
            result4statics['concept_middle_fre'] = get_performance(statics_train['concept_middle_fre'], all_concept_dis)
            result4statics['concept_high_fre'] = get_performance(statics_train['concept_high_fre'], all_concept_dis)
            result4statics['concept_low_acc'] = get_performance(statics_train['concept_low_acc'], all_concept_dis)
            result4statics['concept_middle_acc'] = get_performance(statics_train['concept_middle_acc'], all_concept_dis)
            result4statics['concept_high_acc'] = get_performance(statics_train['concept_high_acc'], all_concept_dis)

            self.objects["logger"].info(
                f"\nconcept frequency\n"
                f"low frequency ({result4statics['concept_low_fre']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_low_fre']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_low_fre']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_low_fre']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_low_fre']['MAE']:<9.6}\n"
                f"middle frequency ({result4statics['concept_middle_fre']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_middle_fre']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_middle_fre']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_middle_fre']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_middle_fre']['MAE']:<9.6}\n"
                f"high frequency ({result4statics['concept_high_fre']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_high_fre']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_high_fre']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_high_fre']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_high_fre']['MAE']:<9.6}\n\n"
                f"concept accuracy\n"
                f"low accuracy ({result4statics['concept_low_acc']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_low_acc']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_low_acc']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_low_acc']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_low_acc']['MAE']:<9.6}\n"
                f"middle accuracy ({result4statics['concept_middle_acc']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_middle_acc']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_middle_acc']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_middle_acc']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_middle_acc']['MAE']:<9.6}\n"
                f"high accuracy ({result4statics['concept_high_acc']['num_sample']:<9} samples), "
                f"AUC: {result4statics['concept_high_acc']['AUC']:<9.6}, "
                f"ACC: {result4statics['concept_high_acc']['ACC']:<9.6}, "
                f"RMSE: {result4statics['concept_high_acc']['RMSE']:<9.6}, "
                f"MAE: {result4statics['concept_high_acc']['MAE']:<9.6}\n"
            )

            # 同时考虑习题和知识点
            all_qc_dis = defaultdict(list)
            for q_id, c_id, p, g in zip(question_all, concept_all, predict_score_all, ground_truth_all):
                all_qc_dis[f"{q_id}_{c_id}"].append((p, g))
            result4statics['qc_low_fre'] = get_performance_qc(statics_train['question_low_fre'], statics_train['concept_low_fre'], all_qc_dis)
            result4statics['qc_high_fre'] = get_performance_qc(statics_train['question_high_fre'], statics_train['concept_high_fre'], all_qc_dis)
            result4statics['qc_low_acc'] = get_performance_qc(statics_train['question_low_acc'], statics_train['concept_low_acc'], all_qc_dis)
            result4statics['qc_high_acc'] = get_performance_qc(statics_train['question_high_acc'], statics_train['concept_high_acc'], all_qc_dis)

            self.objects["logger"].info(
                f"question & concept frequency\n"
                f"low frequency ({result4statics['qc_low_fre']['num_sample']:<9} samples), "
                f"AUC: {result4statics['qc_low_fre']['AUC']:<9.6}, "
                f"ACC: {result4statics['qc_low_fre']['ACC']:<9.6}, "
                f"RMSE: {result4statics['qc_low_fre']['RMSE']:<9.6}, "
                f"MAE: {result4statics['qc_low_fre']['MAE']:<9.6}\n"
                f"high frequency ({result4statics['qc_high_fre']['num_sample']:<9} samples), "
                f"AUC: {result4statics['qc_high_fre']['AUC']:<9.6}, "
                f"ACC: {result4statics['qc_high_fre']['ACC']:<9.6}, "
                f"RMSE: {result4statics['qc_high_fre']['RMSE']:<9.6}, "
                f"MAE: {result4statics['qc_high_fre']['MAE']:<9.6}\n\n"
                f"question & concept accuracy\n"
                f"low accuracy ({result4statics['qc_low_acc']['num_sample']:<9} samples), "
                f"AUC: {result4statics['qc_low_acc']['AUC']:<9.6}, "
                f"ACC: {result4statics['qc_low_acc']['ACC']:<9.6}, "
                f"RMSE: {result4statics['qc_low_acc']['RMSE']:<9.6}, "
                f"MAE: {result4statics['qc_low_acc']['MAE']:<9.6}\n"
                f"high accuracy ({result4statics['qc_high_acc']['num_sample']:<9} samples), "
                f"AUC: {result4statics['qc_high_acc']['AUC']:<9.6}, "
                f"ACC: {result4statics['qc_high_acc']['ACC']:<9.6}, "
                f"RMSE: {result4statics['qc_high_acc']['RMSE']:<9.6}, "
                f"MAE: {result4statics['qc_high_acc']['MAE']:<9.6}\n"
            )

    def evaluate_base_question4multi_concept(self):
        # 按照PYKT的思路实现的，具体见KTDataset
        model = self.objects["models"]["kt_model"]
        data_loader = self.objects["data_loaders"]["test_loader"]
        test_result = Evaluator.evaluate_kt_dataset_base_question4multi_concept(model, data_loader)
        self.objects["logger"].info(
            f"{get_now_time()} test result base question for multi concept dataset\n"
            f"average result is AUC {test_result['average']['AUC']:<9.6}, "
            f"ACC: {test_result['average']['ACC']:<9.6}, "
            f"RMSE: {test_result['average']['RMSE']:<9.6}, "
            f"MAE: {test_result['average']['MAE']:<9.6}\n"
            f"lowest result is AUC: {test_result['lowest']['AUC']:<9.6}, "
            f"ACC: {test_result['lowest']['ACC']:<9.6}, "
            f"RMSE: {test_result['lowest']['RMSE']:<9.6}, "
            f"MAE: {test_result['lowest']['MAE']:<9.6}"
        )

    @staticmethod
    def evaluate_kt_dataset_base_question4multi_concept(model, data_loader):
        # 按照PYKT的思路实现的，具体见KTDataset
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            interaction_index_seq_all = []
            for batch in data_loader:
                predict_score = model.forward4question_evaluate(batch)
                correct_seq = batch["correct_seq"]
                mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
                ground_truth = correct_seq[mask4last.bool()]
                predict_score_all.append(predict_score.detach().cpu().numpy())
                ground_truth_all.append(ground_truth.detach().cpu().numpy())
                interaction_index_seq_all.append(batch["interaction_index_seq"].detach().cpu().numpy())
        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        interaction_index_seq_all = np.concatenate(interaction_index_seq_all, axis=0)
        predict_score_average, predict_score_lowest, ground_truth_new = \
            Evaluator.cal_metric4question(predict_score_all, ground_truth_all, interaction_index_seq_all)
        predict_label_average_all = [1 if p >= 0.5 else 0 for p in predict_score_average]
        predict_label_lowest_all = [1 if p >= 0.5 else 0 for p in predict_score_lowest]
        AUC_average = roc_auc_score(y_true=ground_truth_new, y_score=predict_score_average)
        AUC_lowest = roc_auc_score(y_true=ground_truth_new, y_score=predict_score_lowest)
        ACC_average = accuracy_score(y_true=ground_truth_new, y_pred=predict_label_average_all)
        ACC_lowest = accuracy_score(y_true=ground_truth_new, y_pred=predict_label_lowest_all)
        MAE_average = mean_absolute_error(y_true=ground_truth_new, y_pred=predict_score_average)
        MAE_lowest = mean_absolute_error(y_true=ground_truth_new, y_pred=predict_score_lowest)
        RMSE_average = mean_squared_error(y_true=ground_truth_new, y_pred=predict_score_average) ** 0.5
        RMSE_lowest = mean_squared_error(y_true=ground_truth_new, y_pred=predict_score_lowest) ** 0.5
        return {
            "average": {
                "AUC": AUC_average,
                "ACC": ACC_average,
                "MAE": MAE_average,
                "RMSE": RMSE_average
            },
            "lowest": {
                "AUC": AUC_lowest,
                "ACC": ACC_lowest,
                "MAE": MAE_lowest,
                "RMSE": RMSE_lowest
            }
        }

    @staticmethod
    def cal_metric4question(predict_score, ground_truth, interaction_index):
        # 计算late fusion的指标，包括average和lowest
        score_average = predict_score[0]
        score_lowest = predict_score[0]
        num_concept = 1
        predict_score_average, predict_score_lowest = [], []
        ground_truth_new = []
        last_ground_truth = ground_truth[0]
        idx = interaction_index[0]
        for _, elements in enumerate(zip(predict_score[1:], ground_truth[1:], interaction_index[1:])):
            if idx != elements[2]:
                # 说明是新的一道习题，而不是一道习题的多个知识点
                score_average = score_average / num_concept
                predict_score_average.append(score_average)
                predict_score_lowest.append(score_lowest)
                ground_truth_new.append(last_ground_truth)
                score_average = elements[0]
                score_lowest = elements[0]
                last_ground_truth = elements[1]
                num_concept = 1
                idx = elements[2]
            else:
                score_average += elements[0]
                if elements[0] < score_lowest:
                    score_lowest = elements[0]
                num_concept += 1
        return predict_score_average, predict_score_lowest, ground_truth_new
