import math
import torch

from copy import deepcopy
from collections import defaultdict

from ..util.parse import cal_accuracy4data, get_high_low_accuracy_seqs, cal_diff


class LPKTPlusUtil:
    """
    建模学生学习和遗忘行为

    根据数据集量化学生在各个时刻对各知识点的掌握程度，遵守以下原则：

    （1）某个知识点的习题正确率越高，对该知识点的掌握程度越高

    （2）当连续两次做相同知识点的习题时间间隔较长时，最新一次的影响权重应更大（因为没有学生学习的数据，只有做题数据，所以不能判断这段时间里学生是否学习了相关知识，加大当前记录的权重）

    （3）做对一道题带来的知识点掌握程度提升应该比做错大

    （4）做对一道难题带来的知识点掌握程度提升应该比做做对一道简单题大（这里的难度比较仅限于同一知识点下）

    （5）学生对某个知识点的遗忘程度因随着做（该知识点）题频次增加而下降，甚至考虑学生已经永久掌握该知识点（LPKT的一个缺陷就是建模出来的学生遗忘太快，并且没考虑永久掌握）
    """
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.dataset_train = None
        pass

    def label_user_ability(self, dataset):
        """
        对于一个user的序列，首先提取出一条这样的因子序列，即:\n
        [..., (t, Ct, diff_q, Nc, Sc, last_correct_c, last_interval_c), (t+1, Ct, diff_q, Nc, Sc, last_correct_c, last_interval_c), ...]\n
        来表示认知因子序列，其中t是序列的索引，Ct是t时刻做的习题对应的知识点，diff_q是该习题的难度，Nc是该知识点总共做了多少次（包括当前），Sc是该知识点做对了多少次（包括当前），
        last_correct_c是上一次做该知识点的习题对错情况，last_interval_c是上一次做该知识点的习题距离当前过了多久时间

        然后根据这个因子序列，在提取出user在每个时刻对指定知识点的认知标签和权重，即对于一个user，要提取出这样一条序列:\n
        [..., (t, Ct, master_score_c, weight_c_t), ...]\n

        :param dataset:
        :return:
        """
        factor_seqs = []
        que_difficulty = self.objects["lpkt_plus"]["que_difficulty"]
        for item_data in dataset:
            factor_seq = []
            concept_record = {}
            for t in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][t]
                q_diff = que_difficulty.get(q_id, -1)
                correct = item_data["correct_seq"][t]
                time = item_data["time_seq"][t]
                c_ids = self.objects["data"]["question2concept"][q_id]
                for c_id in c_ids:
                    concept_record.setdefault(c_id, {"N": 0, "S": 0, "last_correct": 0, "last_time": 0})
                    concept_record[c_id]["N"] += 1
                    concept_record[c_id]["S"] += correct
                    concept_record[c_id]["last_correct"] = correct
                    last_time = concept_record[c_id]["last_time"]
                    last_interval_time = -1 if (last_time == 0) else (time - last_time)
                    concept_record[c_id]["last_time"] = time
                    factor_item = (t, c_id, q_diff, concept_record[c_id]["N"], concept_record[c_id]["S"],
                                   concept_record[c_id]["last_correct"], last_interval_time)
                    factor_seq.append(factor_item)
            factor_seqs.append(factor_seq)

        print("")

    def cal_question_diff(self, dataset):
        min_count2drop = self.params["other"]["lpkt_plus"]["min_fre4diff"]
        corrects = defaultdict(int)
        counts = defaultdict(int)

        for item_data in dataset:
            for i in range(item_data["seq_len"]):
                k_id = item_data["question_seq"][i]
                correct = item_data["correct_seq"][i]
                corrects[k_id] += correct
                counts[k_id] += 1

        # 丢弃练习次数少于min_count次的习题或者知识点
        all_ids = list(counts.keys())
        for k_id in all_ids:
            if counts[k_id] < min_count2drop:
                del counts[k_id]
                del corrects[k_id]

        return {k_id: corrects[k_id] / float(counts[k_id]) for k_id in corrects}

    @staticmethod
    def cal_que_discrimination(data_uniformed, params):
        # 公式：总分最高的PERCENT_THRESHOLD学生（H）和总分最低的PERCENT_THRESHOLD学生（L），计算H和L对某道题的通过率，之差为区分度
        NUM2DROP4QUESTION = params.get("min_fre4disc", 30)
        MIN_SEQ_LEN = params.get("min_seq_len4disc", 20)
        PERCENT_THRESHOLD = params.get("percent_threshold", 0.37)

        dataset_question = deepcopy(data_uniformed)
        cal_accuracy4data(dataset_question)
        H_question, L_question = get_high_low_accuracy_seqs(dataset_question, MIN_SEQ_LEN, PERCENT_THRESHOLD)
        H_question_diff = cal_diff(H_question, "question_seq", NUM2DROP4QUESTION)
        L_question_diff = cal_diff(L_question, "question_seq", NUM2DROP4QUESTION)

        intersection_H_L = set(H_question_diff.keys()).intersection(set(L_question_diff.keys()))
        res = {}
        for q_id in intersection_H_L:
            res[q_id] = H_question_diff[q_id] - L_question_diff[q_id]

        return res

    def get_user_proj_weight_init_value(self, concept_accuracy):
        def concept_inti_mastery(acc):
            return math.exp(10 * (acc - 1))

        def get_weight_init_value(y):
            return math.log(y / (1 - y))

        concept_init_value = {}
        num_concept = len(concept_accuracy)
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]["dim_k"]
        min_acc = min(set(concept_accuracy.values()) - {-1})
        unknown_init = concept_inti_mastery(min_acc)
        for c_id, c_acc in concept_accuracy.items():
            concept_init_value[c_id] = concept_inti_mastery(c_acc) if c_acc != -1 else unknown_init

        result = torch.ones(num_concept, dim_emb).float().to(self.params["device"])
        for c_id, init_v in concept_init_value.items():
            x = get_weight_init_value(init_v) / dim_emb
            result[c_id, :] = x

        self.objects["lpkt_plus"]["user_proj_weight_init_value"] = result
