import torch
import numpy as np
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from .BaseModel4CL import BaseModel4CL
from .Module.EncoderLayer import EncoderLayer
from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.MLP import MLP4LLM_emb
from .util import get_mask4last_or_penultimate, parse_question_zero_shot


class SimpleKT(nn.Module, BaseModel4CL):
    model_name = "SimpleKT"

    def __init__(self, params, objects):
        super(SimpleKT, self).__init__()
        super(nn.Module, self).__init__(params, objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        difficulty_scalar = encoder_config["difficulty_scalar"]
        num_concept = encoder_config["num_concept"]
        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dim_final_fc2 = encoder_config["dim_final_fc2"]
        separate_qa = encoder_config["separate_qa"]
        dropout = encoder_config["dropout"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        self.embed_question_difficulty = self.get_embed_question_diff()
        self.embed_concept_variation = nn.Embedding(num_concept, dim_model)
        self.embed_concept = self.get_embed_concept()
        self.embed_interaction_variation = nn.Embedding(2, dim_model)
        if separate_qa:
            # 直接用一个embedding表示在所有concept的interaction
            self.embed_interaction = nn.Embedding(2 * num_concept, dim_model)
        else:
            # 只表示interaction，具体到concept，用concept embedding加interaction embedding表示这个concept的interaction
            self.embed_interaction = nn.Embedding(2, dim_model)

        if use_LLM_emb4question:
            dim_LLM_emb = self.embed_question_difficulty.weight.shape[1]
            if difficulty_scalar:
                self.MLP4question = MLP4LLM_emb(dim_LLM_emb, 1, 0.1)
            else:
                self.MLP4question = MLP4LLM_emb(dim_LLM_emb, dim_model, 0.1)
        else:
            self.reset()
        if use_LLM_emb4concept:
            dim_LLM_emb = self.embed_concept_variation.weight.shape[1]
            self.MLP4concept_variation = MLP4LLM_emb(dim_LLM_emb, dim_model, 0.1)
            self.MLP4concept = MLP4LLM_emb(dim_LLM_emb, dim_model, 0.1)

        self.encoder_layer = EncoderLayer(params, objects)

        self.predict_layer = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, dim_final_fc2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc2, 1),
            nn.Sigmoid()
        )

        # 解析q table
        self.question_head4zero = None
        self.embed_question_difficulty4zero = None
        self.embed_question4zero = None
        self.embed_interaction4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.objects["data"]["question2concept"],
                                                               self.objects["data"]["concept2question"])

    def get_embed_question_diff(self):
        difficulty_scalar = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["difficulty_scalar"]
        num_question = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["num_question"]
        dim_model = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["dim_model"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]

        if use_LLM_emb4question:
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
            embed_question_difficulty = torch.tensor(embed_question, dtype=torch.float).to(self.params["device"])
            embed = nn.Embedding(num_question, embed_question_difficulty.shape[1], _weight=embed_question_difficulty)
            embed.weight.requires_grad = self.params["train_LLM_emb"]
        elif difficulty_scalar:
            # 题目难度用一个标量表示
            embed = nn.Embedding(num_question, 1)
        else:
            # 题目难度用一个embedding表示
            embed = nn.Embedding(num_question, dim_model)

        return embed

    def get_embed_concept(self):
        num_concept = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["num_concept"]
        dim_model = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["dim_model"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        if use_LLM_emb4concept:
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
        else:
            embed = nn.Embedding(num_concept, dim_model)

        return embed

    def reset(self):
        num_question = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["num_question"]
        for p in self.parameters():
            # 这一步至关重要，没有执行constant_初始化直接掉点 也就是一开始必须初始化所有习题难度（向量）为0
            if p.size(0) == num_question:
                nn.init.constant_(p, 0.)

    def get_concept_emb_all(self):
        return self.embed_concept.weight

    def get_concept_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept(batch["concept_seq"])

        return concept_emb

    def get_concept_variation_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept_variation,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept_variation(batch["concept_seq"])

        return concept_emb

    def base_emb(self, batch):
        separate_qa = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["separate_qa"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = self.get_concept_emb(batch)
        if use_LLM_emb4concept:
            concept_emb = self.MLP4concept(concept_emb)
        if separate_qa:
            interaction_seqs = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seqs)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb

        return concept_emb, interaction_emb

    def forward(self, batch):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.get_concept_variation_emb(batch)
        # mu_{q_t}
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        if use_LLM_emb4question:
            question_difficulty_emb = self.MLP4question(question_difficulty_emb)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        # e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}
        interaction_emb = (interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb))

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        latent = self.encoder_layer(encoder_input)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_latent(self, batch, use_emb_dropout=False, dropout=0.1):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb = self.get_concept_emb(batch)
        if use_emb_dropout:
            concept_emb = torch.dropout(concept_emb, dropout, self.training)
        if separate_qa:
            interaction_seqs = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seqs)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.get_concept_variation_emb(batch)
        if use_emb_dropout:
            concept_variation_emb = torch.dropout(concept_variation_emb, dropout, self.training)
        # mu_{q_t}
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        # e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}
        interaction_emb = (
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb))

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        seq_representation = encoder_config.get("seq_representation", "encoder_output")
        if seq_representation == "knowledge_encoder_output":
            latent = self.encoder_layer.get_latent(encoder_input)
        else:
            latent = self.encoder_layer(encoder_input)

        return latent

    def get_latent_last(self, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent(batch, use_emb_dropout, dropout)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean(self, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent(batch, use_emb_dropout, dropout)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def set_emb4zero(self):
        """
        transfer head to tail use gaussian distribution
        :return:
        """
        # difficulty_scalar = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["difficulty_scalar"]
        head2tail_transfer_method = self.params["transfer_head2zero"]["transfer_method"]
        data_type = self.params["datasets_config"]["data_type"]
        indices = []
        tail_qs_emb = []
        for z_q, head_qs in self.question_head4zero.items():
            head_question_indices = torch.tensor(head_qs).long().to(self.params["device"])
            head_qs_emb = self.embed_question_difficulty(head_question_indices).detach().clone()
            if len(head_qs) == 0:
                continue
            indices.append(z_q)
            if head2tail_transfer_method == "gaussian_fit":
                # todo: 这段代码没有检查
                if len(head_qs_emb) > 100:
                    head_qs_emb = head_qs_emb.detach().cpu().numpy()
                    if data_type == "only_question":
                        # 多知识点数据集
                        n_com = 2
                    else:
                        # 单知识点数据集
                        n_com = 1
                    gmm = GaussianMixture(n_components=n_com, random_state=self.params["seed"])
                    gmm.fit(head_qs_emb)
                    gmm_samples = gmm.sample(1)
                    tail_q_emb = torch.from_numpy(gmm_samples[0][0]).item()
                elif len(head_qs_emb) == 0:
                    tail_q_emb = self.embed_question_difficulty.weight.mean().detach().clone()
                else:
                    tail_q_emb = head_qs_emb.mean().detach().detach().clone()
            elif head2tail_transfer_method == "mean_pool":
                # todo: 这段代码没有检查过当question diff emb为标量时是否正确
                tail_q_emb = head_qs_emb.mean(dim=0).unsqueeze(0)
            else:
                raise NotImplementedError()
            tail_qs_emb.append(tail_q_emb)
        indices = torch.tensor(indices)
        tail_qs_emb = torch.cat(tail_qs_emb, dim=0)
        embed_question_difficulty = self.embed_question_difficulty.weight.detach().clone()
        embed_question_difficulty[indices] = tail_qs_emb
        self.embed_question_difficulty4zero = nn.Embedding(
            embed_question_difficulty.shape[0],
            embed_question_difficulty.shape[1],
            _weight=embed_question_difficulty
        )

    def get_predict_score4question_zero(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.get_concept_variation_emb(batch)
        # mu_{q_t}
        question_difficulty_emb = self.embed_question_difficulty4zero(question_seq)
        if use_LLM_emb4question:
            question_difficulty_emb = self.MLP4question(question_difficulty_emb)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        # e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}
        interaction_emb = (
                    interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb))

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        latent = self.encoder_layer(encoder_input)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        if self.params["use_sample_weight"]:
            weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(),
                                                              ground_truth.double(),
                                                              weight=weight)
        else:
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]
