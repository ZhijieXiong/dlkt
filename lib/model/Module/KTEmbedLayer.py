import numpy as np
import torch
import torch.nn as nn


class KTEmbedLayer(nn.Module):
    def __init__(self, params, objects):
        super(KTEmbedLayer, self).__init__()
        self.params = params
        self.objects = objects

        emb_config = self.params["models_config"]["kt_model"]["kt_embed_layer"]
        use_LLM_emb4question = params.get("use_LLM_emb4question", False)
        use_LLM_emb4concept = params.get("use_LLM_emb4concept", False)
        self.emb_dict = {}
        for k, v in emb_config.items():
            if k == "concept":
                self.embed_concept = self.init_concept_with_LLM() if use_LLM_emb4concept else nn.Embedding(v[0], v[1])
                self.emb_dict["concept"] = self.embed_concept
            if k == "question":
                self.embed_question = self.init_question_with_LLM() if use_LLM_emb4question else nn.Embedding(v[0], v[1])
                self.emb_dict["question"] = self.embed_question
            if k == "correct":
                self.embed_correct = nn.Embedding(v[0], v[1])
                self.emb_dict["correct"] = self.embed_correct
            if k == "interaction":
                self.embed_interaction = nn.Embedding(v[0], v[1])
                self.emb_dict["interaction"] = self.embed_interaction

    def init_question_with_LLM(self):
        LLM_question_embeddings = self.objects["data"]["LLM_question_embeddings"]
        all_embeddings = np.array([emb for emb in LLM_question_embeddings.values()])
        mean_embedding = all_embeddings.mean(axis=0).tolist()
        q_id2original_c_id = self.objects["data"]["q_id2original_c_id"]
        data_type = self.params["datasets_config"]["data_type"]

        embed_question = []
        if data_type == "only_question":
            pass
        else:
            for c_id in q_id2original_c_id:
                if str(c_id) in LLM_question_embeddings.keys():
                    embed_question.append(LLM_question_embeddings[str(c_id)])
                else:
                    embed_question.append(mean_embedding)
        embed_question = torch.tensor(embed_question, dtype=torch.float).to(self.params["device"])
        embed = nn.Embedding(embed_question.shape[0], embed_question.shape[1], _weight=embed_question)
        embed.weight.requires_grad = self.params["train_LLM_emb"]
        return embed

    def init_concept_with_LLM(self):
        LLM_concept_embeddings = self.objects["data"]["LLM_concept_embeddings"]
        all_embeddings = np.array([emb for emb in LLM_concept_embeddings.values()])
        mean_embedding = all_embeddings.mean(axis=0).tolist()
        c_id2original_c_id = self.objects["data"]["c_id2original_c_id"]
        data_type = self.params["datasets_config"]["data_type"]

        embed_concept = []
        if data_type == "only_question":
            pass
        else:
            for i in range(len(c_id2original_c_id)):
                c_id = c_id2original_c_id[i]
                if str(c_id) in LLM_concept_embeddings.keys():
                    embed_concept.append(LLM_concept_embeddings[str(c_id)])
                else:
                    embed_concept.append(mean_embedding)

        embed_concept = torch.tensor(embed_concept, dtype=torch.float).to(self.params["device"])
        embed = nn.Embedding(embed_concept.shape[0], embed_concept.shape[1], _weight=embed_concept)
        embed.weight.requires_grad = self.params["train_LLM_emb"]
        return embed

    def get_emb(self, emb_name, emb_index):
        assert self.emb_dict[emb_name] is not None, f"Embedding of {emb_name} is not initialized"
        return self.emb_dict[emb_name](emb_index)

    def get_emb_all(self, emb_name):
        assert self.emb_dict[emb_name] is not None, f"Embedding of {emb_name} is not initialized"
        return self.emb_dict[emb_name].weight

    def get_emb_concatenated(self, seq_names2cat, emb_indices2cat):
        """
        获取拼接后的emb，seq_names2cat是拼接的顺序，emb_indices2cat是id序列（bs * seq_len）
        :param seq_names2cat:
        :param emb_indices2cat:
        :return:
        """
        result = self.get_emb(seq_names2cat[0], emb_indices2cat[0])
        for i, seq in enumerate(seq_names2cat[1:]):
            result = torch.cat((result, self.get_emb(seq, emb_indices2cat[i+1])), dim=-1)
        return result

    def get_emb_question_with_concept_fused(self, question_seq, fusion_type="mean"):
        """
        多知识点embedding融合，如一道多知识点习题的知识点embedding取平均值作为该习题的embedding，再拼接上习题embedding
        :param question_seq:
        :param fusion_type:
        :return:
        """
        emb_question = self.get_emb("question", question_seq)
        emb_concept = self.get_emb("concept", self.objects["data"]["q2c_table"][question_seq])
        mask_concept = self.objects["data"]["q2c_mask_table"][question_seq]
        if fusion_type == "mean":
            emb_concept_fusion = (emb_concept * mask_concept.unsqueeze(-1)).sum(-2)
            emb_concept_fusion = emb_concept_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return torch.cat((emb_concept_fusion, emb_question), dim=-1)

    def get_question2concept_mask(self, question_seq):
        return self.objects["data"]["q2c_table"][question_seq], self.objects["data"]["q2c_mask_table"][question_seq]

    def get_concept_fused_emb(self, question_seq, fusion_type="mean"):
        """
        多知识点embedding融合，如一道多知识点习题的知识点embedding取平均值作为该习题的embedding
        :param question_seq:
        :param fusion_type:
        :return:
        """
        concept_emb = self.get_emb("concept", self.objects["data"]["q2c_table"][question_seq])
        mask_concept = self.objects["data"]["q2c_mask_table"][question_seq]
        if fusion_type == "mean":
            concept_fusion_emb = (concept_emb * mask_concept.unsqueeze(-1)).sum(-2)
            concept_fusion_emb = concept_fusion_emb / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return concept_fusion_emb

    def get_interaction_fused_emb(self, question_seq, correct_seq, num_concept, fusion_type="mean"):
        concept_seq = self.objects["data"]["q2c_table"][question_seq]
        interaction_seq = concept_seq + num_concept * correct_seq.unsqueeze(dim=-1)
        interaction_emb = self.get_emb("interaction", interaction_seq)
        mask_concept = self.objects["data"]["q2c_mask_table"][question_seq]
        if fusion_type == "mean":
            interaction_emb_fusion = (interaction_emb * mask_concept.unsqueeze(-1)).sum(-2)
            interaction_emb_fusion = interaction_emb_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return interaction_emb_fusion

    def get_c_from_q(self, q_id):
        return self.objects["data"]["question2concept"][q_id]

    def get_q_from_c(self, c_id):
        return self.objects["data"]["concept2question"][c_id]

    @staticmethod
    def parse_Q_table(Q_table, device):
        """
        生成多知识点embedding融合需要的数据
        :return:
        """
        question2concept_table = []
        question2concept_mask_table = []
        num_max_c_in_q = np.max(np.sum(Q_table, axis=1))
        num_question = Q_table.shape[0]
        for i in range(num_question):
            cs = np.argwhere(Q_table[i] == 1).reshape(-1).tolist()
            pad_len = num_max_c_in_q - len(cs)
            question2concept_table.append(cs + [0] * pad_len)
            question2concept_mask_table.append([1] * len(cs) + [0] * pad_len)
        question2concept_table = torch.tensor(question2concept_table).long().to(device)
        question2concept_mask_table = torch.tensor(question2concept_mask_table).long().to(device)
        return question2concept_table, question2concept_mask_table, num_max_c_in_q

    @staticmethod
    def concept_fused_emb(embed_concept, q2c_table, q2c_mask_table, question_seq, fusion_type="mean"):
        """
        多知识点embedding融合，如一道多知识点习题的知识点embedding取平均值作为该习题的embedding
        :param embed_concept:
        :param q2c_table:
        :param q2c_mask_table:
        :param question_seq:
        :param fusion_type:
        :return:
        """
        emb_concept = embed_concept(q2c_table[question_seq])
        mask_concept = q2c_mask_table[question_seq]
        if fusion_type == "mean":
            emb_concept_fusion = (emb_concept * mask_concept.unsqueeze(-1)).sum(-2)
            emb_concept_fusion = emb_concept_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return emb_concept_fusion

    @staticmethod
    def other_fused_emb(embed, q2c_table, q2c_mask_table, question_seq, other_table, fusion_type="mean"):
        """
        其它（和concept关联，但是不能直接由concept id取出）embedding的融合，如diff embedding，可能由K个concept，但是diff embedding有k个，
        K > k，因为多个知识点可能diff相同

        :param embed:
        :param q2c_table:
        :param q2c_mask_table:
        :param question_seq:
        :param other_table: K * 1 tensor，其中K是concept的数量，里面元素是concept id和other id的对应关系，即other_table[k]是第k个concept对应的other id
        :param fusion_type:
        :return:
        """
        concept_seq = q2c_table[question_seq]
        emb_other = embed(other_table[concept_seq])
        mask_concept = q2c_mask_table[question_seq]
        if fusion_type == "mean":
            emb_other_fusion = (emb_other * mask_concept.unsqueeze(-1)).sum(-2)
            emb_other_fusion = emb_other_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return emb_other_fusion
