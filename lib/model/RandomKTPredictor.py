import random

import torch
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error

from ..util.statics import cal_frequency, cal_accuracy
from ..util.data import dataset_delete_pad


class RandomKTPredictor:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.train_statics = {
            "question": {},
            "concept": {}
        }
        self.parse_train_set()

    def parse_train_set(self):
        Q_table_single_concept = self.objects["data"]["Q_table_single_concept"]
        num_question = Q_table_single_concept.shape[0]
        num_concept = Q_table_single_concept.shape[1]
        dataset_train = dataset_delete_pad(self.objects["random_kt_predictor"]["dataset_train"])
        question2concept = self.objects["data"]["question2concept_single_concept"]
        for item_data in dataset_train:
            item_data["concept_seq"] = list(map(lambda q: question2concept[q][0], item_data["question_seq"]))

        question_fre = cal_frequency(dataset_train, num_question, target="question")
        question_acc = cal_accuracy(dataset_train, num_question, target="question")
        concept_fre = cal_frequency(dataset_train, num_concept, target="concept")
        concept_acc = cal_accuracy(dataset_train, num_concept, target="concept")

        for q_id in question_fre.keys():
            self.train_statics["question"][q_id] = {
                "fre": question_fre[q_id],
                "acc": question_acc[q_id]
            }

        for c_id in concept_fre.keys():
            self.train_statics["concept"][c_id] = {
                "fre": concept_fre[c_id],
                "acc": concept_acc[c_id]
            }

    def random_predict(self, question_seq, correct_seq, mask_seq):
        question2concept = self.objects["data"]["question2concept_single_concept"]
        question_dict = self.train_statics["question"]
        concept_dict = self.train_statics["concept"]
        max_context_seq_len = self.params["random_kt_predictor"]["max_context_seq_len"]
        weight_context = self.params["random_kt_predictor"]["weight_context"]
        weight_concept = self.params["random_kt_predictor"]["weight_concept"]

        predict_result = []
        history_correct = [correct_seq[0]]
        for i, m in enumerate(mask_seq):
            if i == 0:
                continue

            if m == 0:
                break

            q_id = question_seq[i]
            correct = correct_seq[i]
            c_id = question2concept[q_id][0]

            q_acc = question_dict[q_id]["acc"]
            q_fre = question_dict[q_id]["fre"]
            if q_fre > 10:
                predict_q_score = q_acc
            elif q_fre == 0:
                predict_q_score = -1
            else:
                # 上下浮动，q fre越小，浮动越大
                predict_q_score = min(0.99999, max(0.00001, q_acc + (random.random() * 0.2 - 0.1) * (10 - q_fre) / 10))

            c_acc = concept_dict[c_id]["acc"]
            c_fre = concept_dict[c_id]["fre"]
            if c_fre > 100:
                predict_c_score = c_acc
            elif c_fre == 0:
                predict_c_score = -1
            else:
                # 上下浮动，q fre越小，浮动越大
                predict_c_score = min(0.99999, max(0.00001, c_acc + (random.random() * 0.2 - 0.1) * (100 - q_fre) / 100))

            if i <= max_context_seq_len:
                context_acc = sum(history_correct) / len(history_correct)
            else:
                history_correct_ = history_correct[-max_context_seq_len:]
                context_acc = sum(history_correct_) / len(history_correct_)

            if (predict_q_score == -1) and (predict_c_score == -1):
                # 习题和知识点在训练集中都没有出现过：只依赖历史正确率
                predict_score = context_acc + random.random() * 0.2 - 0.1
            elif predict_q_score == -1:
                # 习题在训练集中未出现过
                predict_score = context_acc * weight_context + (1 - weight_context) * predict_c_score
            else:
                predict_score = context_acc * weight_context + (1 - weight_context) * \
                                (weight_concept * predict_c_score + (1 - weight_concept) * predict_q_score)
            predict_score = min(0.99999, max(0.00001, predict_score + random.random() * 0.1 - 0.05))
            predict_result.append(predict_score)

            history_correct.append(correct)

        return predict_result

    def evaluate_valid(self):
        dataset_valid = self.objects["random_kt_predictor"]["dataset_valid"]
        predict_score = []
        ground_truth = []
        for item_data in dataset_valid:
            q_seq = item_data["question_seq"]
            c_seq = item_data["correct_seq"]
            m_seq = item_data["mask_seq"]
            predict_score += self.random_predict(q_seq, c_seq, m_seq)
            ground_truth += c_seq[1:item_data["seq_len"]]
        predict_label = [1 if p_score > 0.5 else 0 for p_score in predict_score]

        AUC = roc_auc_score(y_true=ground_truth, y_score=predict_score)
        ACC = accuracy_score(y_true=ground_truth, y_pred=predict_label)
        RMSE = mean_squared_error(y_true=ground_truth, y_pred=predict_score)
        MAE = mean_absolute_error(y_true=ground_truth, y_pred=predict_score)

        print(f"valid performance is AUC: {AUC:<9.5}, ACC: {ACC:<9.5}, RMSE: {RMSE:<9.5}, MAE: {MAE:<9.5}")

    def evaluate_test(self):
        dataset_test = self.objects["random_kt_predictor"]["dataset_test"]
        predict_score = []
        ground_truth = []
        for item_data in dataset_test:
            q_seq = item_data["question_seq"]
            c_seq = item_data["correct_seq"]
            m_seq = item_data["mask_seq"]
            predict_score += self.random_predict(q_seq, c_seq, m_seq)
            ground_truth += c_seq[1:item_data["seq_len"]]
        predict_label = [1 if p_score > 0.5 else 0 for p_score in predict_score]

        AUC = roc_auc_score(y_true=ground_truth, y_score=predict_score)
        ACC = accuracy_score(y_true=ground_truth, y_pred=predict_label)
        RMSE = mean_squared_error(y_true=ground_truth, y_pred=predict_score)
        MAE = mean_absolute_error(y_true=ground_truth, y_pred=predict_score)

        print(f"test performance is AUC: {AUC:<9.5}, ACC: {ACC:<9.5}, RMSE: {RMSE:<9.5}, MAE: {MAE:<9.5}")

    def get_predict_score(self, batch):
        question_seq = batch["question_seq"].detach().cpu().numpy().tolist()
        correct_seq = batch["correct_seq"].detach().cpu().numpy().tolist()
        mask_seq = batch["mask_seq"].detach().cpu().numpy().tolist()

        predict_score = []
        for q_seq, c_seq, m_seq in zip(question_seq, correct_seq, mask_seq):
            predict_score += self.random_predict(q_seq, c_seq, m_seq)

        return torch.tensor(predict_score).to(self.params["device"])

    def get_predict_score_seq_len_minus1(self, batch):
        question_seq = batch["question_seq"].detach().cpu().numpy().tolist()
        correct_seq = batch["correct_seq"].detach().cpu().numpy().tolist()
        mask_seq = batch["mask_seq"].detach().cpu().numpy().tolist()

        predict_score_batch = []
        max_seq_len = len(question_seq[0])
        for q_seq, c_seq, m_seq in zip(question_seq, correct_seq, mask_seq):
            predict_score = self.random_predict(q_seq, c_seq, m_seq)
            seq_len = len(predict_score)
            predict_score += [0] * (max_seq_len - seq_len - 1)
            predict_score_batch.append(predict_score)

        return torch.tensor(predict_score_batch).to(self.params["device"])

    def train(self):
        pass

    def eval(self):
        pass
