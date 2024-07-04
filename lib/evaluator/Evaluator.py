import json
import os.path

from collections import defaultdict

from .util import *
from ..util.basic import get_now_time
from ..model.util import get_mask4last_or_penultimate


class Evaluator:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    def evaluate(self):
        transfer_head2zero = self.params.get("transfer_head2zero", {"use_transfer": False})
        use_transfer = transfer_head2zero["use_transfer"]
        fine_grain_config = self.params["evaluate"]["fine_grain"]
        max_seq_len = fine_grain_config["max_seq_len"]
        seq_len_absolute = fine_grain_config["seq_len_absolute"]
        data_type = self.params["datasets_config"]["data_type"]
        all_label_dis, all_score_dis = [[] for _ in range(max_seq_len - 1)], [[] for _ in range(max_seq_len - 1)]

        model = self.objects["models"]["kt_model"]
        data_loader = self.objects["data_loaders"]["test_loader"]
        model.eval()
        with torch.no_grad():
            # 下面4个是用mask select后的格式，即(all_item)
            predict_score_all = []
            ground_truth_all = []
            question_all = []
            concept_all = []
            # result_all_batch是batch格式，即(num_batch * batch_size, seq_len)
            result_all_batch = []
            if use_transfer and hasattr(model, "set_emb4zero"):
                model.set_emb4zero()
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                question_seq = batch["question_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)

                # 用于计算模型在不同序列长度上的效果
                if hasattr(model, "get_predict_score_seq_len_minus1"):
                    predict_score_seq_len_minus1 = model.get_predict_score_seq_len_minus1(batch)
                    result_all_batch.append({
                        "question_seqs": question_seq[:, 1:].detach().cpu().numpy().tolist(),
                        "label_seqs": correct_seq[:, 1:].detach().cpu().numpy().tolist(),
                        "predict_score_seqs": predict_score_seq_len_minus1.detach().cpu().numpy().tolist(),
                        "mask_seqs": batch["mask_seq"][:, 1:].detach().cpu().numpy().tolist()
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
        self.objects["logger"].info(
            f"overall performance is AUC: {AUC:<9.5}, ACC: {ACC:<9.5}, RMSE: {RMSE:<9.5}, MAE: {MAE:<9.5}\n"
        )

        # correlation between history accuracy and model predict score
        window_lens = [10, 15, 20]
        his_acc_ths = [0.4, 0.3, 0.2]
        self.objects["logger"].info(
            f"PPMCC between history average accuracy and current model predict score"
        )
        for his_acc_th in his_acc_ths:
            ppmcc = cal_PPMCC_his_acc_and_cur_model_pred(result_all_batch, window_lens, his_acc_th)
            for window_len in window_lens:
                for k in ["all", "easy", "normal", "hard", "unbalanced"]:
                    self.objects["logger"].info(
                        f"({window_len}, {his_acc_th}), PPMCC between history acc and prediction of {k}"
                        f" is {ppmcc[window_len][k]:.4}"
                    )

        # CORE evaluate (question bias)
        core_evaluation1 = evaluate_core(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), True)
        # core_evaluation2 = evaluate_core(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), False)
        self.print_performance(
            f"\nCORE metric allow repeat ({core_evaluation1['num_sample']:<9}), performance is ", core_evaluation1
        )
        # self.print_performance(
        #     f"CORE metric disallow repeat ({core_evaluation2['num_sample']:<9}), performance is ", core_evaluation2
        # )

        # performance by seq len
        if hasattr(model, "get_predict_score_seq_len_minus1"):
            label_dis4len, score_dis4len, indices4len = evaluate4seq_len(all_label_dis, all_score_dis, seq_len_absolute)
            self.objects["logger"].info("\nevaluation result of samples at different position")
            for i in range(len(label_dis4len)):
                if len(label_dis4len[i]) == 0:
                    continue
                g = np.array(label_dis4len[i])
                p = np.array(score_dis4len[i])
                p_label = [1 if _ >= 0.5 else 0 for _ in p]
                answer_acc = g.sum() / len(g)
                self.objects["logger"].info(
                    f"({indices4len[i][0]:<3}, {indices4len[i][1]:<3}), num of samples: {g.size:<10}, "
                    f"acc of answer: {answer_acc * 100:<4.3}%, performance is "
                    f"AUC: {roc_auc_score(y_true=g, y_score=p):<9.5}, "
                    f"ACC: {accuracy_score(g, p_label):<9.5}, "
                    f"RMSE: {mean_squared_error(y_true=g, y_pred=p) ** 0.5:<9.5}, "
                    f"MAE: {mean_absolute_error(y_true=g, y_pred=p):<9.5}"
                )

        # 测试集的偏差子集
        if hasattr(model, "get_predict_score_seq_len_minus1"):
            self.objects["logger"].info("\nevaluation result of seq biased samples")
            his_window_lens = [10, 15, 20]
            acc_ths = [0.4, 0.3, 0.2]
            for his_window_len in his_window_lens:
                for acc_th in acc_ths:
                    seq_fine_grained_performance = get_seq_fine_grained_performance(
                        result_all_batch, his_window_len, acc_th
                    )

                    self.print_performance(
                        f"({his_window_len}, {acc_th}) seq easy samples "
                        f"({seq_fine_grained_performance['easy']['num_sample']:<9}), performance is ",
                        seq_fine_grained_performance['easy']
                    )
                    self.print_performance(
                        f"({his_window_len}, {acc_th}) seq normal samples "
                        f"({seq_fine_grained_performance['normal']['num_sample']:<9}), performance is ",
                        seq_fine_grained_performance['normal']
                    )
                    self.print_performance(
                        f"({his_window_len}, {acc_th}) seq hard samples "
                        f"({seq_fine_grained_performance['hard']['num_sample']:<9}), performance is ",
                        seq_fine_grained_performance['hard']
                    )
                    self.print_performance(
                        f"({his_window_len}, {acc_th}) cold start samples "
                        f"({seq_fine_grained_performance['cold_start']['num_sample']:<9}), performance is ",
                        seq_fine_grained_performance['cold_start']
                    )

        train_statics_common_path = fine_grain_config["train_statics_common_path"]
        if not os.path.exists(train_statics_common_path):
            return

        with open(train_statics_common_path, "r") as f:
            train_statics_common = json.load(f)
        question_acc_dict = {}
        for q_id, q_acc in train_statics_common["question_acc"].items():
            question_acc_dict[int(q_id)] = q_acc
        train_statics_common["question_acc"] = question_acc_dict
        if "concept_acc" in train_statics_common.keys():
            concept_acc_dict = {}
            for c_id, c_acc in train_statics_common["concept_acc"].items():
                concept_acc_dict[int(c_id)] = c_acc
            train_statics_common["concept_acc"] = concept_acc_dict

        his_acc_ths = [0.4, 0.3, 0.2]
        self.objects["logger"].info(
            f"\nPPMCC between question accuracy in train dataset and current model predict score"
        )
        for his_acc_th in his_acc_ths:
            ppmcc = cal_PPMCC_train_question_acc_and_cur_model_pred(result_all_batch, train_statics_common, his_acc_th)
            for k in ["all", "easy", "normal", "hard", "unbalanced"]:
                self.objects["logger"].info(
                    f"({his_acc_th}), PPMCC between question acc and prediction of {k} is {ppmcc[k]:.4}"
                )

        acc_ths = [0.4, 0.3, 0.2]
        self.objects["logger"].info("\nevaluation result of question biased samples")
        for acc_th in acc_ths:
            question_fine_grained_performance = get_question_fine_grained_performance(
                result_all_batch, train_statics_common, acc_th
            )

            self.print_performance(
                f"({acc_th}) question easy samples "
                f"({question_fine_grained_performance['easy']['num_sample']:<9}), performance is ",
                question_fine_grained_performance['easy']
            )
            self.print_performance(
                f"({acc_th}) question normal samples "
                f"({question_fine_grained_performance['normal']['num_sample']:<9}), performance is ",
                question_fine_grained_performance['normal']
            )
            self.print_performance(
                f"({acc_th}) question hard samples "
                f"({question_fine_grained_performance['hard']['num_sample']:<9}), performance is ",
                question_fine_grained_performance['hard']
            )
            self.print_performance(
                f"({acc_th}) question unseen samples "
                f"({question_fine_grained_performance['unseen']['num_sample']:<9}), performance is ",
                question_fine_grained_performance['unseen']
            )

        if hasattr(model, "get_predict_score_seq_len_minus1"):
            his_window_lens = [10, 15, 20]
            acc_ths = [0.4, 0.3, 0.2]
            self.objects["logger"].info("\nevaluation result of double biased samples")
            for his_window_len in his_window_lens:
                for acc_th in acc_ths:
                    double_fine_grained_performance = get_double_fine_grained_performance(
                        result_all_batch, train_statics_common, his_window_len, acc_th
                    )

                    self.print_performance(
                        f"({his_window_len}, {acc_th}) double easy samples "
                        f"({double_fine_grained_performance['easy']['num_sample']:<9}), performance is ",
                        double_fine_grained_performance['easy']
                    )
                    self.print_performance(
                        f"({his_window_len}, {acc_th}) double hard samples "
                        f"({double_fine_grained_performance['hard']['num_sample']:<9}), performance is ",
                        double_fine_grained_performance['hard']
                    )

        # 高|中|低频习题|知识点的性能；高|中|低正确率习题|知识点的性能
        train_statics_special_path = fine_grain_config["train_statics_special_path"]
        if not os.path.exists(train_statics_special_path):
            return

        result4statics = {}
        with open(train_statics_special_path, "r") as f:
            train_statics_special = json.load(f)

        question_all = np.concatenate(question_all, axis=0)
        all_question_dis = defaultdict(list)
        for q_id, p, g in zip(question_all, predict_score_all, ground_truth_all):
            all_question_dis[q_id].append((p, g))

        if model.objects["data"].get("question_no_head_qs", False):
            # performance of zero shot questions of not-head questions sharing one concept
            performance4zero_q = get_performance(model.objects["data"].get("question_no_head_qs"), all_question_dis)
            self.objects["logger"].info(
                f"\nzero shot questions of no head questions sharing one concept"
            )
            self.print_performance(f"zero shot ({performance4zero_q['num_sample']:<9} samples), ", performance4zero_q)
        else:
            self.objects["logger"].info("")

        result4statics['question_zero_fre'] = get_performance(train_statics_special['question_zero_fre'], all_question_dis)
        result4statics['question_low_fre'] = get_performance(train_statics_special['question_low_fre'], all_question_dis)
        result4statics['question_middle_fre'] = get_performance(train_statics_special['question_middle_fre'], all_question_dis)
        result4statics['question_high_fre'] = get_performance(train_statics_special['question_high_fre'], all_question_dis)
        result4statics['question_low_acc'] = get_performance(train_statics_special['question_low_acc'], all_question_dis)
        result4statics['question_middle_acc'] = get_performance(train_statics_special['question_middle_acc'], all_question_dis)
        result4statics['question_high_acc'] = get_performance(train_statics_special['question_high_acc'], all_question_dis)

        self.print_performance(f"question zero shot ({result4statics['question_zero_fre']['num_sample']:<9} samples), ",
                               result4statics['question_zero_fre'])
        self.print_performance(f"question low frequency ({result4statics['question_low_fre']['num_sample']:<9} samples), ",
                               result4statics['question_low_fre'])
        self.print_performance(f"question middle frequency ({result4statics['question_middle_fre']['num_sample']:<9} samples), ",
                               result4statics['question_middle_fre'])
        self.print_performance(f"question high frequency ({result4statics['question_high_fre']['num_sample']:<9} samples), ",
                               result4statics['question_high_fre'])

        # self.print_performance(f"question low accuracy ({result4statics['question_low_acc']['num_sample']:<9} samples), ",
        #                        result4statics['question_low_acc'])
        # self.print_performance(f"question middle accuracy ({result4statics['question_middle_acc']['num_sample']:<9} samples), ",
        #                        result4statics['question_middle_acc'])
        # self.print_performance(f"question high accuracy ({result4statics['question_high_acc']['num_sample']:<9} samples), ",
        #                        result4statics['question_high_acc'])

        # if data_type != "only_question":
        #     concept_all = np.concatenate(concept_all, axis=0)
        #     all_concept_dis = defaultdict(list)
        #     for c_id, p, g in zip(concept_all, predict_score_all, ground_truth_all):
        #         all_concept_dis[c_id].append((p, g))
        #     result4statics['concept_low_fre'] = get_performance(train_statics_special['concept_low_fre'], all_concept_dis)
        #     result4statics['concept_middle_fre'] = get_performance(train_statics_special['concept_middle_fre'], all_concept_dis)
        #     result4statics['concept_high_fre'] = get_performance(train_statics_special['concept_high_fre'], all_concept_dis)
        #     result4statics['concept_low_acc'] = get_performance(train_statics_special['concept_low_acc'], all_concept_dis)
        #     result4statics['concept_middle_acc'] = get_performance(train_statics_special['concept_middle_acc'], all_concept_dis)
        #     result4statics['concept_high_acc'] = get_performance(train_statics_special['concept_high_acc'], all_concept_dis)
        #
        #     self.print_performance(f"concept low frequency ({result4statics['concept_low_fre']['num_sample']:<9} samples), ",
        #                            result4statics['concept_low_fre'])
        #     self.print_performance(f"concept middle frequency ({result4statics['concept_middle_fre']['num_sample']:<9} samples), ",
        #                            result4statics['concept_middle_fre'])
        #     self.print_performance(f"concept high frequency ({result4statics['concept_high_fre']['num_sample']:<9} samples), ",
        #                            result4statics['concept_high_fre'])
        #
        #     self.print_performance(f"concept low accuracy ({result4statics['concept_low_acc']['num_sample']:<9} samples), ",
        #                            result4statics['concept_low_acc'])
        #     self.print_performance(f"concept middle accuracy ({result4statics['concept_middle_acc']['num_sample']:<9} samples), ",
        #                            result4statics['concept_middle_acc'])
        #     self.print_performance(f"concept high accuracy ({result4statics['concept_high_acc']['num_sample']:<9} samples), ",
        #                            result4statics['concept_high_acc'])
        #
        #     # 同时考虑习题和知识点
        #     all_qc_dis = defaultdict(list)
        #     for q_id, c_id, p, g in zip(question_all, concept_all, predict_score_all, ground_truth_all):
        #         all_qc_dis[f"{q_id}_{c_id}"].append((p, g))
        #     result4statics['qc_low_fre'] = get_performance_qc(train_statics_special['question_low_fre'],
        #                                                       train_statics_special['concept_low_fre'], all_qc_dis)
        #     result4statics['qc_high_fre'] = get_performance_qc(train_statics_special['question_high_fre'],
        #                                                        train_statics_special['concept_high_fre'], all_qc_dis)
        #     result4statics['qc_low_acc'] = get_performance_qc(train_statics_special['question_low_acc'],
        #                                                       train_statics_special['concept_low_acc'], all_qc_dis)
        #     result4statics['qc_high_acc'] = get_performance_qc(train_statics_special['question_high_acc'],
        #                                                        train_statics_special['concept_high_acc'], all_qc_dis)
        #
        #     self.print_performance(f"qc low frequency ({result4statics['qc_low_fre']['num_sample']:<9} samples), ",
        #                            result4statics['qc_low_fre'])
        #     self.print_performance(f"qc high frequency ({result4statics['qc_high_fre']['num_sample']:<9} samples), ",
        #                            result4statics['qc_high_fre'])
        #
        #     self.print_performance(f"qc low accuracy ({result4statics['qc_low_acc']['num_sample']:<9} samples), ",
        #                            result4statics['qc_low_acc'])
        #     self.print_performance(f"qc high accuracy ({result4statics['qc_high_acc']['num_sample']:<9} samples), ",
        #                            result4statics['qc_high_acc'])

    def print_performance(self, prefix, performance_dict):
        self.objects["logger"].info(
            f"{prefix}"
            f"AUC: {performance_dict['AUC']:<9.6}, "
            f"ACC: {performance_dict['ACC']:<9.6}, "
            f"RMSE: {performance_dict['RMSE']:<9.6}, "
            f"MAE: {performance_dict['MAE']:<9.6}"
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
