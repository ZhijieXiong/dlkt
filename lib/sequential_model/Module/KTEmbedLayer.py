import numpy as np
import torch
import torch.nn as nn


from lib.util import parse


class KTEmbedLayer(nn.Module):
    def __init__(self, params, objects):
        super(KTEmbedLayer, self).__init__()
        self.params = params
        self.objects = objects

        emb_config = self.params["models_config"]["kt_model"]["kt_embed_layer"]
        self.emb_dict = {}
        for k, v in emb_config.items():
            if len(v) > 0:
                if k == "concept":
                    self.embed_concept = nn.Embedding(v[0], v[1])
                    self.emb_dict["concept"] = self.embed_concept
                if k == "question":
                    self.embed_question = nn.Embedding(v[0], v[1])
                    self.emb_dict["question"] = self.embed_question
                if k == "correct":
                    self.embed_correct = nn.Embedding(v[0], v[1])
                    self.emb_dict["correct"] = self.embed_correct
                if k == "interaction":
                    self.embed_interaction = nn.Embedding(v[0], v[1])
                    self.emb_dict["interaction"] = self.embed_interaction

        self.Q_table = self.objects["data"].get("Q_table", None)
        # 如果有Q table的话，question2concept_table和question2concept_mask_table都是(num_q, num_max_c)的tensor
        # num_max_c表示在该数据集中一道习题最多对应几个知识点
        self.question2concept_table = None
        self.question2concept_mask_table = None
        # 同理，如果有Q table的话，可以分析出question2concept_list和concept2question_list
        # 前者index表示习题id，value表示该习题对应的知识点列表
        # 后者index表示知识点id，value表示该知识点对应的习题列表
        self.question2concept_list = None
        self.concept2question_list = None
        self.num_max_concept = None
        if self.Q_table is not None:
            self.parse_Q_table()

    def get_emb(self, emb_name, emb_index):
        assert self.emb_dict[emb_name] is not None, f"Embedding of {emb_name} is not initialized"
        return self.emb_dict[emb_name](emb_index)

    def get_emb_all(self, emb_name):
        assert self.emb_dict[emb_name] is not None, f"Embedding of {emb_name} is not initialized"
        return self.emb_dict[emb_name].weight

    def get_emb_concatenated(self, seq_names2cat, emb_indices2cat):
        """
        获取拼接后的emb
        :param seq_names2cat:
        :param emb_indices2cat:
        :return:
        """
        result = self.get_emb(seq_names2cat[0], emb_indices2cat[0])
        for i, seq in enumerate(seq_names2cat[1:]):
            result = torch.cat((result, self.get_emb(seq, emb_indices2cat[i+1])), dim=-1)
        return result

    def parse_Q_table(self):
        """
        用于多知识点embedding融合，如对于一道多知识点习题的知识点embedding取平均值作为该习题的embedding
        :return:
        """
        self.question2concept_list = parse.question2concept_from_Q(self.Q_table)
        self.concept2question_list = parse.concept2question_from_Q(self.Q_table)

        device = self.params["device"]
        question2concept_table = []
        question2concept_mask_table = []
        num_max_c_in_q = np.max(np.sum(self.Q_table, axis=1))
        num_question = self.Q_table.shape[0]
        for i in range(num_question):
            cs = np.argwhere(self.Q_table[i] == 1).reshape(-1).tolist()
            pad_len = num_max_c_in_q - len(cs)
            question2concept_table.append(cs + [0]*pad_len)
            question2concept_mask_table.append([1] * len(cs) + [0]*pad_len)
        self.question2concept_table = torch.tensor(question2concept_table).long().to(device)
        self.question2concept_mask_table = torch.tensor(question2concept_mask_table).long().to(device)
        self.num_max_concept = num_max_c_in_q

    def get_emb_question_with_concept_fused(self, question_seq, concept_fusion="mean"):
        # 对于多知识点数据集，获取拼接了知识点emb（以某种方式融合）的习题emb
        emb_question = self.get_emb("question", question_seq)
        emb_concept = self.get_emb("concept", self.question2concept_table[question_seq])
        mask_concept = self.question2concept_mask_table[question_seq]
        if concept_fusion == "mean":
            emb_concept_fusion = (emb_concept * mask_concept.unsqueeze(-1)).sum(-2)
            emb_concept_fusion = emb_concept_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return torch.cat((emb_concept_fusion, emb_question), dim=-1)

    def get_question2concept_mask(self, question_seq):
        return self.question2concept_table[question_seq], self.question2concept_mask_table[question_seq]

    def get_concept_fused_emb(self, question_seq, concept_fusion="mean"):
        # 对于多知识点数据集，获取拼接了知识点emb（以某种方式融合）的习题emb
        emb_concept = self.get_emb("concept", self.question2concept_table[question_seq])
        mask_concept = self.question2concept_mask_table[question_seq]
        if concept_fusion == "mean":
            emb_concept_fusion = (emb_concept * mask_concept.unsqueeze(-1)).sum(-2)
            emb_concept_fusion = emb_concept_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return emb_concept_fusion

    def get_c_from_q(self, q_id):
        return self.question2concept_list[q_id]

    def get_q_from_c(self, c_id):
        return self.concept2question_list[c_id]
