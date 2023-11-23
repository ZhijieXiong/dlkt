import torch
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from .util import record_dis4seq_len, evaluate4seq_len


class Evaluator:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

    def evaluate(self):
        evaluate_config = self.params["evaluate"]["fine_grain"]
        max_seq_len = evaluate_config["max_seq_len"]
        seq_len_absolute = evaluate_config["seq_len_absolute"]
        seq_len_percent = evaluate_config["seq_len_percent"]
        # base_type = self.params["datasets_config"]["test"]["kt"]["base_type"]
        all_label_dis, all_score_dis = [[] for _ in range(max_seq_len - 1)], [[] for _ in range(max_seq_len - 1)]

        model = self.objects["models"]["kt_model"]
        data_loader = self.objects["data_loaders"]["test_loader"]
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                correct_seq = batch["correct_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                predict_score_seq_len_minus1 = model.get_predict_score_seq_len_minus1(batch)
                label_dis, score_dis = record_dis4seq_len(correct_seq[:, 1:],
                                                          predict_score_seq_len_minus1,
                                                          batch["mask_seq"][:, 1:])
                for i in range(max_seq_len-1):
                    all_score_dis[i] += score_dis[i]
                    all_label_dis[i] += label_dis[i]
                predict_score = model.get_predict_score(batch).detach().cpu().numpy()
                ground_truth = torch.masked_select(correct_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            predict_label_all = [1 if p >= 0.5 else 0 for p in predict_score_all]
            AUC = roc_auc_score(y_true=ground_truth_all, y_score=predict_score_all)
            ACC = accuracy_score(y_true=ground_truth_all, y_pred=predict_label_all)
            MAE = mean_absolute_error(y_true=ground_truth_all, y_pred=predict_score_all)
            RMSE = mean_squared_error(y_true=ground_truth_all, y_pred=predict_score_all) ** 0.5

        print(f"AUC: {AUC:<9.5}, ACC: {ACC:<9.5}, RMSE: {MAE:<9.5}, MAE: {RMSE:<9.5}")
        evaluate4seq_len(all_label_dis, all_score_dis, seq_len_absolute, seq_len_percent)
