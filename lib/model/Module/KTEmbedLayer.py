import numpy as np
import torch
import json
import torch.nn as nn


class KTEmbedLayer(nn.Module):
    def __init__(self, params, objects):
        super(KTEmbedLayer, self).__init__()
        self.params = params
        self.objects = objects

        emb_config = self.params["models_config"]["kt_model"]["kt_embed_layer"]
        self.emb_dict = {}
        for k, v in emb_config.items():
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

    def get_emb_concatenated_w_net(self, seq_names2cat, emb_indices2cat, nets):
        """
        获取拼接后的emb，其中emb会通过一个网络进行转换，seq_names2cat是拼接的顺序，emb_indices2cat是id序列（bs * seq_len）
        :param seq_names2cat:
        :param emb_indices2cat:
        :param nets:
        :return:
        """
        result = nets[0](self.get_emb(seq_names2cat[0], emb_indices2cat[0]))
        for i, seq in enumerate(seq_names2cat[1:]):
            result = torch.cat((result, nets[i](self.get_emb(seq, emb_indices2cat[i+1]))), dim=-1)
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

    def get_emb_question_with_concept_fused_w_nets(self, question_seq, q_net, c_net, fusion_type="mean"):
        """
        多知识点embedding融合，其中emb会通过一个网络进行转换，如一道多知识点习题的知识点embedding取平均值作为该习题的embedding，再拼接上习题embedding
        :param question_seq:
        :param fusion_type:
        :param q_net:
        :param c_net:
        :return:
        """
        emb_question = q_net(self.get_emb("question", question_seq))
        emb_concept = c_net(self.get_emb("concept", self.objects["data"]["q2c_table"][question_seq]))
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
    def interaction_fused_emb(embed_concept, q2c_table, q2c_mask_table, question_seq, correct_seq, num_concept,
                              fusion_type="mean"):
        """
        多知识点的交互融合，如一道多知识点习题的知识点作答结果（如DKVMN）取平均值作为该交互的embedding
        :param embed_concept:
        :param q2c_table:
        :param q2c_mask_table:
        :param question_seq:
        :param correct_seq:
        :param num_concept:
        :param fusion_type:
        :return:
        """
        emb_interaction = embed_concept(q2c_table[question_seq] + num_concept * correct_seq.unsqueeze(-1))
        mask_concept = q2c_mask_table[question_seq]
        if fusion_type == "mean":
            emb_interaction_fusion = (emb_interaction * mask_concept.unsqueeze(-1)).sum(-2)
            emb_interaction_fusion = emb_interaction_fusion / mask_concept.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return emb_interaction_fusion

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


class KTEmbedLayer2(nn.Module):
    r"""
    Args:
        embed_configs : dict("embed_name", embed_config)
            "num_item" (int): size of the dictionary of embeddings
            "dim_item" (int): the size of each embedding vector
            "learnable" (bool, optional):
            "embed_path" (str, optional):
            "init_method" (str, optional): "init_correctness_1", "init_correctness_2"
    """
    def __init__(self, embed_configs):
        super(KTEmbedLayer2, self).__init__()
        self.embed_configs = embed_configs

        for embed_name, embed_config in embed_configs.items():
            # must
            num_item = embed_config["num_item"]
            dim_item = embed_config["dim_item"]

            if (("embed_path" not in embed_config) and
                    ("learnable" not in embed_config) and
                    ("init_method" not in embed_config)):
                # 默认nn.Embedding
                self.__setattr__(embed_name, nn.Embedding(num_item, dim_item))
            elif "embed_path" not in embed_config:
                # 根据init_method使用不同初始化方法
                init_method = embed_config["init_method"]
                if init_method in ["init_correctness_1", "init_correctness_2"]:
                    # 初始化correctness，默认是不可学习的
                    self.__setattr__(embed_name, self.init_constant_embed(embed_config))
                else:
                    self.__setattr__(embed_name, nn.Embedding(num_item, dim_item))
                    # 默认是可学习的
                    self.__getattr__(embed_name).weight.requires_grad = embed_config.get("learnable", True)
            else:
                # 默认只要有embed_path，就是合法地址，并且可以成功导入pretrained emb
                self.__setattr__(embed_name, self.init_embed_from_pretrained(embed_name, embed_config))

    @staticmethod
    def init_constant_embed(embed_config):
        """
        初始化固定的embed layer，如下\n
        1、init_correctness_1：(2, dim)，0用全0表示，1用全1表示\n
        2、init_correctness_2：(2, dim)，0用左边一半元素为1，右边一半元素为0表示，1则相反

        :param embed_config:
        :return:
        """
        init_method = embed_config["init_method"]
        dim_item = embed_config["dim_item"]
        if init_method == "init_correctness_1":
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0] = torch.zeros(dim_item)
            embed.weight.data[1] = torch.ones(dim_item)
        elif init_method == "init_correctness_2":
            dim_half = dim_item // 2
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0, :dim_half] = 0
            embed.weight.data[0, dim_half:] = 1
            embed.weight.data[1, :dim_half] = 1
            embed.weight.data[1, dim_half:] = 0
        else:
            raise NotImplementedError()
        embed.weight.requires_grad = False
        return embed

    def init_embed_from_pretrained(self, embed_name, embed_config):
        """

        :param embed_name:
        :param embed_config:
        :return:
        """
        num_item = embed_config["num_item"]
        dim_item = embed_config["dim_item"]
        embed_path = embed_config["embed_path"]

        with open(embed_path, 'r') as f:
            precomputed_embeddings = json.load(f)
        pretrained_emb_tensor = torch.tensor(
            [precomputed_embeddings[str(i)] for i in range(len(precomputed_embeddings))], dtype=torch.float)

        num_emb, dim_emb = pretrained_emb_tensor.shape

        assert num_item == num_emb

        # Normalize the lengths to 1, for convenience.
        norms = pretrained_emb_tensor.norm(p=2, dim=1, keepdim=True)
        pretrained_emb_tensor = pretrained_emb_tensor / norms
        # Now scale to expected size.
        pretrained_emb_tensor = pretrained_emb_tensor * np.sqrt(num_item)

        if dim_item != dim_emb:
            self.__setattr__(f"{embed_name}ProjectionLayer", nn.Linear(dim_emb, dim_item))

        return nn.Embedding.from_pretrained(pretrained_emb_tensor, freeze=not embed_config.get("learnable", True))

    def get_emb(self, embed_name, item_index):
        """
        获取指定embed里的emb

        :param embed_name:
        :param item_index:
        :return:
        """
        embed_config = self.embed_configs[embed_name]

        if "embed_path" not in embed_config:
            return self.__getattr__(embed_name)(item_index)
        else:
            if hasattr(self, f"{embed_name}ProjectionLayer"):
                return self.__getattr__(f"{embed_name}ProjectionLayer")(
                    self.__getattr__(embed_name)(item_index)
                )
            else:
                return self.__getattr__(embed_name)(item_index)

    def get_emb_concatenated(self, cat_order, item_index2cat):
        """
        获取拼接后的emb，cat_order是拼接的顺序，item_index2cat是id序列（bs * seq_len）

        :param cat_order:
        :param item_index2cat:
        :return:
        """
        concatenated_emb = self.get_emb(cat_order[0], item_index2cat[0])
        for i, embed_name in enumerate(cat_order[1:]):
            concatenated_emb = torch.cat((concatenated_emb, self.get_emb(embed_name, item_index2cat[i + 1])), dim=-1)
        return concatenated_emb

    def get_emb_fused1(
            self,
            related_embed_name,
            base2related_transfer_table,
            base2related_mask_table,
            base_item_index,
            fusion_method="mean"
    ):
        """
        获取多个emb融合（如mean pool）后的emb

        例如一道习题关联多个知识点，则首先根据习题id找到对应的多个知识点id，然后取出对应的知识点emb并fuse

        在上面那个例子中，将习题记为base，知识点记为related

        base和related是1对多的关系

        :param related_embed_name:
        :param base2related_transfer_table:
        :param base2related_mask_table:
        :param base_item_index:
        :param fusion_method:
        :return:
        """
        embed_related = self.__getattr__(related_embed_name)
        related_emb = embed_related(base2related_transfer_table[base_item_index])
        mask = base2related_mask_table[base_item_index]
        if fusion_method == "mean":
            related_emb_fusion = (related_emb * mask.unsqueeze(-1)).sum(-2)
            related_emb_fusion = related_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related_emb_fusion

    def get_emb_fused2(
            self,
            related2_embed_name,
            base2related1_transfer_table,
            base2related1_mask_table,
            related_1_to_2_transfer_table,
            base_item_index,
            fusion_method="mean"
    ):
        """
        获取多个emb融合（如mean pool）后的emb

        例如一道习题关联多个知识点，每个知识点又关联一个知识点难度，则首先根据习题id找到对应的多个知识点id，然后取出对应的知识点难度id，
        并取出知识点难度emb，然后fuse

        在上面那个例子中，将习题记为base，知识点记为related1，知识点难度记为related2

        base和related是1对多的关系，related1和related2是1对1的关系

        :param related2_embed_name:
        :param base2related1_transfer_table:
        :param base2related1_mask_table:
        :param related_1_to_2_transfer_table: 以上面例子为基础，假设有K个知识点，k个知识点难度，则该table为K * 1 tensor，
        里面元素是知识点id和知识点难度id的对应关系，即other_table[k]是第k个concept对应的知识点难度id
        :param base_item_index:
        :param fusion_method:
        :return:
        """
        embed_related2 = self.__getattr__(related2_embed_name)
        related1_item_index = base2related1_transfer_table[base_item_index]
        related2_emb = embed_related2(related_1_to_2_transfer_table[related1_item_index])
        mask = base2related1_mask_table[base_item_index]
        if fusion_method == "mean":
            related2_emb_fusion = (related2_emb * mask.unsqueeze(-1)).sum(-2)
            related2_emb_fusion = related2_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related2_emb_fusion