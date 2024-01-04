import math
import os.path
import random
import json
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
        self.data_type = None
        self.question2concept = parse_util.question2concept_from_Q(self.objects["data"]["Q_table"])
        self.concept2question = parse_util.concept2question_from_Q(self.objects["data"]["Q_table"])
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

    def parse(self, data, data_type):
        """
        传入的data需要是去掉padding的
        :param data:
        :param data_type:
        :return:
        """
        self.data_type = data_type
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
        num_question = dataset_config_this["kt4aug"]["informative_aug"]["num_question"]
        self.concept_similarity_table = {c_id: None for c_id in range(num_concept)}
        self.question_similarity_table = {q_id: None for q_id in range(num_question)}

        self.data = data

        if num_concept is not None:
            self.get_concept_similar_table()
        self.get_question_similarity_table()
        self.data = None

    def get_question_similarity_table(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = self.objects["file_manager"].get_setting_dir(setting_name)
        question_similar_table_name = file_name.replace(".txt", "_question_similar_table.json")
        similar_table_path = os.path.join(setting_dir, question_similar_table_name)

        if os.path.exists(similar_table_path):
            question_similarity_table = data_util.load_json(similar_table_path)
            for k in question_similarity_table:
                self.question_similarity_table[int(k)] = question_similarity_table[k]
            return

        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_question = dataset_config_this["kt4aug"]["informative_aug"]["num_question"]

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
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
        C = dict()
        N = dict()
        data = deepcopy(self.data)
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
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
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
        if self.data_type == "multi_concept":
            data = data_util.dataset_agg_concept(self.data)
        else:
            data = self.data
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
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
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
        for i in range(num_concept):
            concepts_related = []
            for j, tup in self.concept_next_candidate[i].items():
                if j == i:
                    continue
                # tup[0]表示j出现在i后面的次数，tup[0]-tup[1]表示j出现在i后面（非相邻）的次数
                concepts_related.append((j, tup[0], tup[0]-tup[1]))
            concepts_related = list(filter(lambda three: three[1] > 100 and three[2] > 10, concepts_related))
            self.concept_similarity_table[i] = [i]
            self.concept_similarity_table[i] += list(map(lambda three: three[0], concepts_related))

    def cal_concept_similarity_table_RCD_graph(self):
        pass

    def get_concept_similar_table(self):
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        informative_config = dataset_config_this["kt4aug"]["informative_aug"]
        setting_name = dataset_config_this["setting_name"]
        file_name = dataset_config_this["file_name"]
        setting_dir = self.objects["file_manager"].get_setting_dir(setting_name)
        offline_sim_type = informative_config["offline_sim_type"]
        concept_similar_table_name = file_name.replace(".txt", f"_concept_similar_table_by_{offline_sim_type}.json")
        similar_table_path = os.path.join(setting_dir, concept_similar_table_name)
        if os.path.exists(similar_table_path):
            concept_similarity_table = data_util.load_json(similar_table_path)
            for k in concept_similarity_table:
                self.concept_similarity_table[int(k)] = concept_similarity_table[k]
            return

        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
        self.concept_frequency, self.low_frequency_c, self.high_frequency_c = self.cal_frequency(
            int(num_concept * 0.8), "concept")
        if offline_sim_type == "difficulty":
            self.cal_concept_similarity_table_difficulty()
        elif offline_sim_type == "ItemCF_IUF":
            self.cal_concept_similarity_table_ItemCF_IUF()
        elif offline_sim_type == "order":
            self.cal_concept_similarity_table_order()
        elif offline_sim_type == "RCD_graph":
            self.cal_concept_similarity_table_RCD_graph()
        else:
            raise NotImplementedError()

        if similar_table_path is not None:
            with open(similar_table_path, "w") as f:
                json.dump(self.concept_similarity_table, f)

    def cal_frequency(self, freq_threshold=30, target="question"):
        # 统计频率，对于低频率的知识点/习题，其难度和区分度赋值为中间水平
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]
        num_question = dataset_config_this["kt4aug"]["informative_aug"]["num_question"]

        target_seq = "concept_seq" if target == "concept" else "question_seq"
        data = self.data
        if self.data_type == "multi_concept" and target == "question":
            data = data_util.dataset_agg_concept(data)
        num_item = num_concept if target == "concept" else num_question
        item_seqs = list(map(lambda item_data: item_data[target_seq], data))
        items = []
        for question_seq in item_seqs:
            items += question_seq
        item_frequency = Counter(items)

        for item in range(num_item):
            if item not in item_frequency.keys():
                item_frequency[item] = 0

        frequency_sorted = list(map(lambda e: e[0],
                                    sorted(list(item_frequency.items()), reverse=True, key=lambda ele: ele[1])))
        low_frequency = set(frequency_sorted[freq_threshold:])
        high_frequency = set(frequency_sorted[:freq_threshold])

        return item_frequency, low_frequency, high_frequency

    def cal_question_difficulty(self):
        self.question_difficulty = []
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_question = dataset_config_this["kt4aug"]["informative_aug"]["num_question"]

        data = self.data
        if self.data_type == "multi_concept":
            data = data_util.dataset_agg_concept(data)
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
        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        num_concept = dataset_config_this["kt4aug"]["informative_aug"]["num_concept"]

        count = {i: 0 for i in range(num_concept)}
        correct = {i: 0 for i in range(num_concept)}
        for item_data in self.data:
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


class OnlineSimilarity:
    def __init__(self):
        self.concept_similarity = None
        self.concept_similarity_table = {}

    def analysis(self, concept_emb):
        num_concept = len(concept_emb)
        self.concept_similarity = cos_sim_self(concept_emb)
        self.concept_similarity += np.diag([1] * num_concept)
        for c in range(num_concept):
            self.concept_similarity_table[c] = np.argsort(self.concept_similarity[:, c])[::-1]
        self.concept_similarity = None

    def get_similar_concept(self, concept_id, top_k=10):
        return self.concept_similarity_table[concept_id][1:top_k + 1]
