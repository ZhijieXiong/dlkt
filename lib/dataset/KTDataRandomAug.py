import random
import math
import numpy as np

from collections import defaultdict
from copy import deepcopy

from ..util import parse


class KTDataRandomAug:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects

        self.easier_concepts = None
        self.harder_concepts = None
        self.easier_questions = None
        self.harder_questions = None
        self.question_difficulty_in_concept = None

    def parse_data(self, data_uniformed):
        self.get_question_difficulty(data_uniformed)
        self.get_concept_difficulty(data_uniformed)

    def get_concept_difficulty(self, data_uniformed):
        data_type = self.params["datasets_config"]["data_type"]

        if data_type == "only_question":
            concept_seqs = []
            correct_seqs = []
            for item_data in data_uniformed:
                seq_len = item_data["seq_len"]
                concept_seq = []
                correct_seq = []
                for q_id, c in zip(item_data["question_seq"][:seq_len], item_data["correct_seq"][:seq_len]):
                    c_ids = self.objects["data"]["question2concept"][q_id]
                    concept_seq += c_ids
                    correct_seq += [c] * len(c_ids)
                concept_seqs.append(concept_seq)
                correct_seqs.append(correct_seq)
        else:
            concept_seqs = [item_data["concept_seq"][:item_data["seq_len"]] for item_data in data_uniformed]
            correct_seqs = [item_data["correct_seq"][:item_data["seq_len"]] for item_data in data_uniformed]

        concept_correct = defaultdict(int)
        concept_count = defaultdict(int)
        for concept_seq, correct_seq in zip(concept_seqs, correct_seqs):
            for c_id, correct in zip(concept_seq, correct_seq):
                concept_correct[c_id] += correct
                concept_count[c_id] += 1

        concept_difficulty = {
            c_id: concept_correct[c_id] / float(concept_count[c_id]) for c_id in concept_correct
        }
        concept_ordered_difficulty = [
            item[0] for item in sorted(concept_difficulty.items(), key=lambda x: x[1])
        ]
        easier_concepts = {}
        harder_concepts = {}
        for i, s in enumerate(concept_ordered_difficulty):
            if i == 0:
                # the hardest
                easier_concepts[s] = concept_ordered_difficulty[i + 1]
                harder_concepts[s] = s
            elif i == len(concept_ordered_difficulty) - 1:
                # the easiest
                easier_concepts[s] = s
                harder_concepts[s] = concept_ordered_difficulty[i - 1]
            else:
                easier_concepts[s] = concept_ordered_difficulty[i + 1]
                harder_concepts[s] = concept_ordered_difficulty[i - 1]
        self.easier_concepts = easier_concepts
        self.harder_concepts = harder_concepts

    def get_question_difficulty(self, data_uniformed):
        # 对每个知识点下的习题进行难度计算
        data_type = self.params["datasets_config"]["data_type"]

        if data_type == "only_question":
            concept_seqs = []
            correct_seqs = []
            question_seqs = []
            for item_data in data_uniformed:
                seq_len = item_data["seq_len"]
                concept_seq = []
                correct_seq = []
                question_seq = []
                for q_id, c in zip(item_data["question_seq"][:seq_len], item_data["correct_seq"][:seq_len]):
                    c_ids = self.objects["data"]["question2concept"][q_id]
                    concept_seq += c_ids
                    correct_seq += [c] * len(c_ids)
                    question_seq += [q_id] * len(c_ids)
                concept_seqs.append(concept_seq)
                correct_seqs.append(correct_seq)
                question_seqs.append(question_seq)
        else:
            concept_seqs = [item_data["concept_seq"][:item_data["seq_len"]] for item_data in data_uniformed]
            correct_seqs = [item_data["correct_seq"][:item_data["seq_len"]] for item_data in data_uniformed]
            question_seqs = [item_data["question_seq"][:item_data["seq_len"]] for item_data in data_uniformed]

        question_in_concept = defaultdict(dict)
        for question_seq, concept_seq, correct_seq in zip(question_seqs, concept_seqs, correct_seqs):
            for q_id, c_id, correct in zip(question_seq, concept_seq, correct_seq):
                question_in_concept[c_id].setdefault(q_id, defaultdict(int))
                question_in_concept[c_id][q_id]["correct"] += correct
                question_in_concept[c_id][q_id]["count"] += 1

        self.question_difficulty_in_concept = {
            c_id: {q_id: (info["correct"] / info["count"]) for q_id, info in q_info.items()}
            for (c_id, q_info) in question_in_concept.items()
        }

        # 仿照知识点，把每个知识点下的习题分成harder和easier，方便替换
        self.easier_questions = {}
        self.harder_questions = {}
        for c_id in self.question_difficulty_in_concept.keys():
            question_difficulty = self.question_difficulty_in_concept[c_id]
            question_ordered_difficulty = [
                item[0] for item in sorted(question_difficulty.items(), key=lambda x: x[1])
            ]
            easier_questions = {}
            harder_questions = {}
            if len(question_ordered_difficulty) == 1:
                easier_questions[question_ordered_difficulty[0]] = question_ordered_difficulty[0]
                harder_questions[question_ordered_difficulty[0]] = question_ordered_difficulty[0]
                continue
            for i, s in enumerate(question_ordered_difficulty):
                if i == 0:
                    # the hardest
                    easier_questions[s] = question_ordered_difficulty[i + 1]
                    harder_questions[s] = s
                elif i == len(question_ordered_difficulty) - 1:
                    # the easiest
                    easier_questions[s] = s
                    harder_questions[s] = question_ordered_difficulty[i - 1]
                else:
                    easier_questions[s] = question_ordered_difficulty[i + 1]
                    harder_questions[s] = question_ordered_difficulty[i - 1]
            self.easier_questions[c_id] = easier_questions
            self.harder_questions[c_id] = harder_questions

    def replace_seq(self, sample, replace_prob):
        """
        先进行知识点替换，再在替换后的知识点下随机选一道习题
        :param sample:
        :param replace_prob:
        :return:
        """
        data_type = self.params["datasets_config"]["data_type"]
        sample = deepcopy(sample)
        seq_len = sample["seq_len"]
        replace_idx = random.sample(list(range(seq_len)), k=max(1, int(seq_len * replace_prob)))
        for i in replace_idx:
            if data_type == "only_question":
                q_id = sample["question_seq"][i]
                c_ids = self.objects["data"]["question2concept"][q_id]
                c_id = random.choice(c_ids)
            else:
                c_id = sample["concept_seq"][i]

            correct = sample["correct_seq"][i]
            if correct == 0 and c_id in self.harder_concepts.keys():
                # if the response is wrong, then replace a skill with the harder one
                similar_c = self.harder_concepts[c_id]
            elif correct == 1 and c_id in self.easier_concepts.keys():
                # if the response is correct, then replace a skill with the easier one
                similar_c = self.easier_concepts[c_id]
            else:
                # easiest and most difficult, only replace question
                similar_c = sample["concept_seq"][i]

            if data_type != "only_question":
                sample["concept_seq"][i] = similar_c

            if "question_seq" in sample.keys():
                similar_q = random.choice(parse.get_question_from_concept(similar_c, self.objects["data"]["Q_table"]))
                sample["question_seq"][i] = similar_q

        return sample

    @staticmethod
    def mask_seq(sample, mask_prob, mask_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < mask_min_seq_len:
            return deepcopy(sample)

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)

        sample_new = deepcopy(sample)
        mask_idx = random.sample(list(range(seq_len)), k=max(1, int(seq_len * mask_prob)))
        for i in mask_idx:
            for k in seq_keys:
                sample_new[k][i] = -1
        for k in seq_keys:
            sample_new[k] = list(filter(lambda x: x != -1, sample_new[k]))
        sample_new["seq_len"] = len(sample_new["correct_seq"])

        return sample_new

    @staticmethod
    def permute_seq(sample, perm_prob, perm_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < perm_min_seq_len:
            return deepcopy(sample)

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)
        reorder_seq_len = max(2, math.floor(perm_prob * seq_len))
        # count和not_permute用于控制while True的循环次数，当循环次数超过一定次数，都没能得到合适的start_pos时，跳出循环，不做置换
        count = 0
        not_permute = False
        while True:
            if count >= 50:
                not_permute = True
                break
            count += 1
            start_pos = random.randint(0, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break
        if not_permute:
            return deepcopy(sample)

        sample_new = deepcopy(sample)
        perm = np.random.permutation(reorder_seq_len)
        for k in seq_keys:
            seq = sample_new[k]
            sample_new[k] = seq[:start_pos] + np.asarray(seq[start_pos:start_pos + reorder_seq_len])[perm]. \
                tolist() + seq[start_pos + reorder_seq_len:]

        return sample_new

    @staticmethod
    def crop_seq(sample, crop_prob, crop_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < crop_min_seq_len:
            return deepcopy(sample)

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)
        cropped_seq_len = min(seq_len - 1, math.floor((1 - crop_prob) * seq_len))
        count = 0
        not_crop = False
        while True:
            if count >= 50:
                not_crop = True
                break
            count += 1
            start_pos = random.randint(0, seq_len - cropped_seq_len)
            if start_pos + cropped_seq_len < seq_len:
                break
        if not_crop:
            return deepcopy(sample)

        sample_new = deepcopy(sample)
        for k in seq_keys:
            sample_new[k] = sample_new[k][start_pos: start_pos + cropped_seq_len]
        sample_new["seq_len"] = len(sample_new["correct_seq"])

        return sample_new

    @staticmethod
    def negative_seq(correct_seq, neg_prob):
        correct_seq_neg = deepcopy(correct_seq)
        seq_len = len(correct_seq)
        negative_idx = random.sample(list(range(seq_len)), k=int(seq_len * neg_prob))
        for i in negative_idx:
            correct_seq_neg[i] = 1 - correct_seq_neg[i]

        return correct_seq_neg
