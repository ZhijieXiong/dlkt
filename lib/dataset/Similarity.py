import math
import os.path
import random
import json
import gc
import torch
import numpy as np

from collections import Counter
from copy import deepcopy

from ..util import data as data_util
from ..util import parse as parse_util


def cos_sim_self(matrix):
    matrix = matrix.detach().clone()
    matrix.requires_grad_(False)
    sim_mat = []
    for i in range(matrix.shape[0]):
        item_sim = torch.cosine_similarity(matrix, matrix[i].view(1, -1)).unsqueeze(-1)
        sim_mat.append(item_sim.detach().cpu().numpy().astype(np.float16))

    sim_mat = np.concatenate(sim_mat, axis=1)
    return sim_mat


class OfflineSimilarity:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

        self.data = None
        self.question2concept = parse_util.question2concept_from_Q(self.objects["data"]["Q_table"])
        self.concept2question = parse_util.concept2question_from_Q(self.objects["data"]["Q_table"])
        self.concept_high_distinction = []
        self.question_high_distinction = []
        self.question_frequency = None
        self.concept_frequency = None
        self.low_frequency_q = None
        self.high_frequency_q = None
        self.low_frequency_c = None
        self.high_frequency_c = None
        self.concept_sim_mat = None
        self.concept_difficulty = None
        self.question_difficulty = None
        self.concept_dissimilarity = None
        self.concept_similarity = None
        self.question_dissimilarity = None
        self.concept_next_candidate = None
        self.concept_similarity_table = None
        self.question_similarity_table = None

    def parse(self, data):
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        num_question = informative_aug_config["num_question"]
        self.concept_similarity_table = {c_id: None for c_id in range(num_concept)}
        self.question_similarity_table = {q_id: None for q_id in range(num_question)}

        self.data = deepcopy(data)

        if num_concept is not None:
            self.get_concept_similar_table()
        self.get_question_similarity_table()
        self.data = None
        gc.collect()

    # def load_distinction_table(self):
    #     distinct_table_path = self.envs.get("distinct_table", None)
    #     if distinct_table_path is not None and os.path.exists(distinct_table_path):
    #         distinct_table = load_json(distinct_table_path)
    #
    #         self.concept_high_distinction = distinct_table["concept"]
    #         for i in range(len(self.concept_high_distinction)):
    #             self.concept_high_distinction[i] = int(self.concept_high_distinction[i])
    #
    #         self.question_high_distinction = distinct_table["question"]
    #         for i in range(len(self.question_high_distinction)):
    #             self.question_high_distinction[i] = int(self.question_high_distinction[i])
    #
    #         return
    #     self.cal_distinction()
    #     if distinct_table_path is not None:
    #         with open(distinct_table_path, "w") as f:
    #             json.dump({"question": self.question_high_distinction, "concept": self.concept_high_distinction}, f)

    def get_question_similarity_table(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = self.objects["file_manager"].get_setting_dir(setting_name)
        question_similar_table_name = file_name.replace(".txt", "_question_similar_table.json")
        similar_table_path = os.path.join(setting_dir, question_similar_table_name)

        if similar_table_path is not None and os.path.exists(similar_table_path):
            question_similarity_table = data_util.load_json(similar_table_path)
            for k in question_similarity_table:
                self.question_similarity_table[int(k)] = question_similarity_table[k]
            return

        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_question = informative_aug_config["num_question"]

        self.question_frequency, self.low_frequency_q, self.high_frequency_q = self.cal_frequency(
            int(num_question * 0.2), "question")
        self.cal_question_difficulty()
        # 计算两个习题之间难度差值的绝对值，作为不相似度
        self.question_dissimilarity = np.abs(
            np.subtract.outer(np.array(self.question_difficulty), np.array(self.question_difficulty)))
        self.question_dissimilarity += np.diag([-2] * num_question)
        # 对于低频的习题，降低它和其它习题的不相似度，增加被选中的概率
        for q in self.low_frequency_q:
            self.question_dissimilarity[q] -= 0.1
        # 存储习题不相似度为table，方便后面查询（习题只存同一知识点下的习题）
        for q in range(num_question):
            cs_correspond = self.question2concept[q]
            qs_share_concept = []
            for c_correspond in cs_correspond:
                qs_share_concept += self.concept2question[c_correspond]
            qs_share_concept = list(set(qs_share_concept))
            dissimilarity_score = self.question_dissimilarity[qs_share_concept, q]
            sort_index = np.argsort(dissimilarity_score)
            self.question_similarity_table[q] = np.array(qs_share_concept)[sort_index].tolist()

        if similar_table_path is not None:
            with open(similar_table_path, "w") as f:
                json.dump(self.question_similarity_table, f)

    def cal_concept_similarity_table_ItemCF_IUF(self):
        # 用推荐系统的公式计算知识点相似度
        data = data_util.dataset_delete_pad(self.data)
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        C = dict()
        N = dict()
        concept_seqs = list(map(lambda item_data: item_data["concept_seq"], data))
        for concept_seq in concept_seqs:
            for i in concept_seq:
                N.setdefault(i, 0)
                N[i] += 1
                for j in concept_seq:
                    C.setdefault(i, {})
                    C[i].setdefault(j, 0)
                    C[i][j] += 1 / math.log(1 + len(concept_seq) * 1.0)
        self.concept_similarity = np.zeros((num_concept, num_concept))
        for c_i in C:
            max_score = max(C[c_i].values())
            min_score = min(C[c_i].values())
            for c_j, score in C[c_i].items():
                self.concept_similarity[c_i][c_j] = (score - min_score) / (max_score - min_score)
        self.concept_similarity += np.diag([10] * num_concept)
        # 对于低频的知识点，增加它和其它知识点的相似度，增加被选中的概率
        for c in self.low_frequency_c:
            self.concept_similarity[c] += 0.1
        for c in range(num_concept):
            self.concept_similarity_table[c] = np.argsort(self.concept_dissimilarity[:, c])[::-1].tolist()

    def cal_concept_similarity_table_difficulty(self):
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        self.cal_concept_difficulty()
        # 计算两个知识点之间难度差值的绝对值，作为不相似度
        self.concept_dissimilarity = np.abs(
            np.subtract.outer(np.array(self.concept_difficulty), np.array(self.concept_difficulty)))
        self.concept_dissimilarity += np.diag([-10] * num_concept)
        # 对于低频的知识点，降低它和其它知识点的不相似度，增加被选中的概率
        for c in self.low_frequency_c:
            self.concept_dissimilarity[c] -= 0.1
        for c in range(num_concept):
            self.concept_similarity_table[c] = np.argsort(self.concept_dissimilarity[:, c]).tolist()

    def concept_order(self):
        # 从训练数据中统计知识点之间先后出现次数，并区分距离远近
        data = data_util.dataset_delete_pad(data_util.dataset_agg_concept(self.data))
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        self.concept_next_candidate = {}
        for i in range(num_concept):
            self.concept_next_candidate[i] = {}
            for j in range(num_concept):
                # self.concept_next_candidate[c][i]对应的2元组，第1个元素表示i在c后面出现的次数，第2个元素表示i离c近（相邻）的次数
                self.concept_next_candidate[i][j] = [0, 0]
        for item_data in data:
            q_first = item_data["question_seq"][0]
            cs_first = self.question2concept[q_first]
            cs_previous = set(cs_first)
            cs_last = cs_first
            for q_cur in item_data["question_seq"][1:]:
                cs_cur = self.question2concept[q_cur]
                for c_previous in cs_previous:
                    for c_cur in cs_cur:
                        self.concept_next_candidate[c_previous][c_cur][0] += 1
                        if c_previous in cs_last:
                            self.concept_next_candidate[c_previous][c_cur][1] += 1
                cs_previous.union(cs_cur)
                cs_last = cs_cur

    def cal_concept_similarity_table_order(self):
        self.concept_order()
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        for i in range(num_concept):
            concepts_related = []
            for j, tup in self.concept_next_candidate[i].items():
                if j == i:
                    continue
                concepts_related.append((j, tup[0], tup[0]-tup[1]))
            # 选出和目标知识点先后出现次数大于10的知识点，先按照先后次数排序，再按照相距较远的次数排序（打破分布）
            concepts_related = list(filter(lambda three: three[1] > 10, concepts_related))
            num_related = int(0.8 * len(concepts_related))
            # 第一次排序选择
            concepts_related = sorted(concepts_related, key=lambda three: three[1], reverse=True)[:num_related]
            # 第二次排序
            concepts_related = sorted(concepts_related, key=lambda three: three[2], reverse=True)
            self.concept_similarity_table[i] = [i]
            self.concept_similarity_table[i] += list(map(lambda three: three[0], concepts_related))

    def get_concept_similar_table(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        informative_config = dataset_config_this["kt4aug"]["informative_aug"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = self.objects["file_manager"].get_setting_dir(setting_name)
        offline_sim_type = informative_config["offline_sim_type"]
        concept_similar_table_name = file_name.replace(".txt", f"_concept_similar_table_by_{offline_sim_type}.json")
        similar_table_path = os.path.join(setting_dir, concept_similar_table_name)
        if similar_table_path is not None and os.path.exists(similar_table_path):
            concept_similarity_table = data_util.load_json(similar_table_path)
            for k in concept_similarity_table:
                self.concept_similarity_table[int(k)] = concept_similarity_table[k]
            return

        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        self.concept_frequency, self.low_frequency_c, self.high_frequency_c = self.cal_frequency(
            int(num_concept * 0.8), "concept")
        if offline_sim_type == "difficulty":
            self.cal_concept_similarity_table_difficulty()
        elif offline_sim_type == "ItemCF_IUF":
            self.cal_concept_similarity_table_ItemCF_IUF()
        elif offline_sim_type == "order":
            self.cal_concept_similarity_table_order()
        else:
            raise NotImplementedError()

        if similar_table_path is not None:
            with open(similar_table_path, "w") as f:
                json.dump(self.concept_similarity_table, f)

    def cal_frequency(self, freq_threshold=30, target="question"):
        # 统计频率，对于低频率的知识点/习题，其难度和区分度赋值为中间水平
        data_type = self.params["datasets_config"]["data_type"]
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]
        num_question = informative_aug_config["num_question"]

        target_seq = "concept_seq" if target == "concept" else "question_seq"
        data = deepcopy(self.data)
        if data_type == "multi_concept" and target == "concept":
            data = data_util.dataset_agg_concept(data)
        data = data_util.dataset_delete_pad(data)
        num_item = num_concept if target == "concept" else num_question
        item_seqs = list(map(lambda item_data: item_data[target_seq], data))
        items = []
        for question_seq in item_seqs:
            items += question_seq
        item_frequency = Counter(items)

        for item in range(num_item):
            if item not in item_frequency.keys():
                item_frequency[item] = 0

        frequency_sorted = list(map(lambda e: e[0], sorted(list(item_frequency.items()), reverse=True, key=lambda ele: ele[1])))
        low_frequency = set(frequency_sorted[freq_threshold:])
        high_frequency = set(frequency_sorted[:freq_threshold])

        return item_frequency, low_frequency, high_frequency

    # def cal_distinction(self, min_seq_len=10):
    #     # 公式：总分最高的27%学生（H）和总分最低的27%学生（L），计算H和L对某道题的通过率，之差为区分度，区分度大于0.4为高区分度习题，低于0.2表示区分度不好
    #     def cal_diff(D, k):
    #         # 计算正确率，习题或者知识点
    #         seqs = [item[k] for item in D]
    #         correct_seqs = [item["correct_seq"] for item in D]
    #         corrects = defaultdict(int)
    #         counts = defaultdict(int)
    #
    #         for seq, correct_seq in zip(seqs, correct_seqs):
    #             for k_id, correct in zip(seq, correct_seq):
    #                 corrects[k_id] += correct
    #                 counts[k_id] += 1
    #
    #         # 丢弃练习次数少于min_count次的习题或者知识点
    #         all_ids = list(counts.keys())
    #         for k_id in all_ids:
    #             if counts[k_id] < min_count_table[self.envs["dataset_name"]]:
    #                 del counts[k_id]
    #                 del corrects[k_id]
    #
    #         return {k_id: corrects[k_id] / float(counts[k_id]) for k_id in corrects}
    #
    #     def get_high_distinction(H, L, update_target):
    #         intersection_H_L = set(H.keys()).intersection(set(L.keys()))
    #         for k_id in intersection_H_L:
    #             if H[k_id] - L[k_id] >= 0.35:
    #                 update_target.append(k_id)
    #
    #     dataset_concept = delete_pad(self.data)
    #     # 统计知识点正确率
    #     accuracy_list = []
    #     count_statics = 0
    #     for item_data in dataset_concept:
    #         seq_len = item_data["seq_len"]
    #         if seq_len < min_seq_len:
    #             continue
    #         num_right = 0
    #         num_wrong = 0
    #         for i, m in enumerate(item_data["mask_seq"]):
    #             if m == 0:
    #                 break
    #             num_right += item_data["correct_seq"][i]
    #             num_wrong += (1 - item_data["correct_seq"][i])
    #         accuracy = num_right / (num_right + num_wrong)
    #         item_data["acc"] = accuracy
    #         accuracy_list.append(accuracy)
    #         count_statics += 1
    #     accuracy_list = sorted(accuracy_list)
    #     high_acc = accuracy_list[int(count_statics * (1 - 0.27))]
    #     low_acc = accuracy_list[int(count_statics * 0.27)]
    #     H_concept = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] >= high_acc, dataset_concept))
    #     L_concept = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] <= low_acc, dataset_concept))
    #     H_concept_diff = cal_diff(H_concept, "concept_seq")
    #     L_concept_diff = cal_diff(L_concept, "concept_seq")
    #     get_high_distinction(H_concept_diff, L_concept_diff, self.concept_high_distinction)
    #
    #     if self.num_question is not None:
    #         if self.envs["dataset_name"] in datasets_multi_concept:
    #             dataset_question = delete_pad(dataset_multi2single(self.data, self.envs["dataset_name"]))
    #         else:
    #             dataset_question = dataset_concept
    #
    #         # 统计习题正确率
    #         if self.envs["dataset_name"] in datasets_multi_concept:
    #             accuracy_list = []
    #             count_statics = 0
    #             for item_data in dataset_question:
    #                 seq_len = item_data["seq_len"]
    #                 if seq_len < min_seq_len:
    #                     continue
    #                 num_right = 0
    #                 num_wrong = 0
    #                 for i, m in enumerate(item_data["mask_seq"]):
    #                     if m == 0:
    #                         break
    #                     num_right += item_data["correct_seq"][i]
    #                     num_wrong += (1 - item_data["correct_seq"][i])
    #                 accuracy = num_right / (num_right + num_wrong)
    #                 item_data["acc"] = accuracy
    #                 accuracy_list.append(accuracy)
    #                 count_statics += 1
    #             accuracy_list = sorted(accuracy_list)
    #             high_acc = accuracy_list[int(count_statics * (1 - 0.27))]
    #             low_acc = accuracy_list[int(count_statics * 0.27)]
    #             H_question = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] >= high_acc, dataset_question))
    #             L_question = list(filter(lambda item: item["seq_len"] >= min_seq_len and item["acc"] <= low_acc, dataset_question))
    #         else:
    #             H_question = H_concept
    #             L_question = L_concept
    #         H_question_diff = cal_diff(H_question, "question_seq")
    #         L_question_diff = cal_diff(L_question, "question_seq")
    #         get_high_distinction(H_question_diff, L_question_diff, self.question_high_distinction)

    def cal_question_difficulty(self):
        self.question_difficulty = []
        data_type = self.params["datasets_config"]["data_type"]
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_question = informative_aug_config["num_question"]

        data = deepcopy(self.data)
        if data_type == "multi_concept":
            data = data_util.dataset_agg_concept(data)
        data = data_util.dataset_delete_pad(data)
        count = {i: 0 for i in range(num_question)}
        correct = {i: 0 for i in range(num_question)}
        for item_data in data:
            for q_id, c in zip(item_data["question_seq"], item_data["correct_seq"]):
                count[q_id] += 1
                if c == 1:
                    correct[q_id] += 1

        for q_id in range(num_question):
            if count[q_id] == 0:
                self.question_difficulty.append(0.5)
                continue
            if q_id in self.low_frequency_q:
                self.question_difficulty.append(0.3 + (0.4 * correct[q_id] / count[q_id]))
            else:
                self.question_difficulty.append(correct[q_id] / count[q_id])

    def cal_concept_difficulty(self):
        self.concept_difficulty = []
        informative_aug_config = self.params["other"]["informative_aug_config"]
        num_concept = informative_aug_config["num_concept"]

        data = data_util.dataset_delete_pad(self.data)
        count = {i: 0 for i in range(num_concept)}
        correct = {i: 0 for i in range(num_concept)}
        for item_data in data:
            for c_id, c in zip(item_data["concept_seq"], item_data["correct_seq"]):
                count[c_id] += 1
                if c == 1:
                    correct[c_id] += 1

        for c_id in range(num_concept):
            if count[c_id] == 0:
                self.concept_difficulty.append(0.5)
                continue
            if c_id in self.low_frequency_c:
                self.concept_difficulty.append(0.3 + (0.4 * correct[c_id] / count[c_id]))
            else:
                self.concept_difficulty.append(correct[c_id] / count[c_id])

    def get_similar_concept(self, concept_id, top_k=10):
        if len(self.concept_similarity_table[concept_id]) > 1:
            return self.concept_similarity_table[concept_id][1:top_k + 1]
        else:
            return self.concept_similarity_table[concept_id]

    def get_similar_question(self, question_id, top_k=10):
        if len(self.question_similarity_table[question_id]) > 1:
            return self.question_similarity_table[question_id][1:top_k + 1]
        else:
            return self.question_similarity_table[question_id]

    def get_qs_in_concept(self, concept_id):
        return self.concept2question[concept_id]

    def get_random_q_in_concept(self, concept_id):
        num_qs = len(self.concept2question[concept_id])
        num_seg = num_qs // 100
        index_selected = int(random.random() * num_seg) + int(random.random() * 100)
        index_selected = min(index_selected, num_qs - 1)
        return self.concept2question[concept_id][index_selected]

    def get_high_distinct_concept(self):
        return self.concept_high_distinction

    def get_high_distinct_question(self):
        concept_correspond = [random.choice(self.question2concept[q_id]) for q_id in self.question_high_distinction]
        return self.question_high_distinction, concept_correspond


class OnlineSimilarity:
    def __init__(self, concept_embed, envs, question_embed=None):
        self.envs = envs
        self.question2concept = parse_util.question2concept_from_Q(self.envs["Q_table"])
        self.concept2question = parse_util.concept2question_from_Q(self.envs["Q_table"])
        self.concept_embed = concept_embed
        self.question_embed = question_embed
        self.concept_similarity = None
        self.question_similarity = None
        self.concept_similarity_table = {}
        self.question_similarity_table = {}

        self.analysis()

    def analysis(self):
        num_concept = len(self.concept_embed)
        self.concept_similarity = cos_sim_self(self.concept_embed)
        self.concept_similarity += np.diag([1] * num_concept)
        for c in range(num_concept):
            self.concept_similarity_table[c] = np.argsort(self.concept_similarity[:, c])[::-1]
        self.concept_similarity = None

        if self.question_embed is not None:
            num_question = len(self.question_embed)
            self.question_similarity = cos_sim_self(self.question_embed)
            self.question_similarity += np.diag([1] * num_question)
            for q in range(num_question):
                c_correspond = random.choice(self.question2concept[q])
                qs_share_concept = self.concept2question[c_correspond]
                similarity_score = self.question_similarity[qs_share_concept, q]
                sort_index = np.argsort(similarity_score)[::-1]
                self.question_similarity_table[q] = np.array(qs_share_concept)[sort_index]
            self.question_similarity = None

        gc.collect()

    def re_analysis(self, concept_embed, question_embed=None):
        self.concept_embed = concept_embed
        self.question_embed = question_embed
        self.analysis()

    def get_similar_concept(self, concept_id, top_k=10):
        return self.concept_similarity_table[concept_id][1:top_k + 1]

    def get_similar_question(self, question_id, top_k=10):
        if len(self.question_similarity_table[question_id]) > 1:
            return self.question_similarity_table[question_id][1:top_k + 1]
        else:
            return self.question_similarity_table[question_id]
