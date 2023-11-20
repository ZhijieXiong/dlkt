import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .Module.EncoderLayer import EncoderLayer
from .loss_util import duo_info_nce, binary_entropy
from .util import get_mask4last_or_penultimate


class AKT(nn.Module):
    def __init__(self, params, objects):
        super(AKT, self).__init__()
        self.params = params
        self.objects = objects

        # embed init
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        dim_emb = encoder_config["dim_model"]
        separate_qa = encoder_config["separate_qa"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        self.embed_question_difficulty = nn.Embedding(num_question, 1)
        self.embed_concept_variation = nn.Embedding(num_concept, dim_emb)
        self.embed_interaction_variation = nn.Embedding(2 * num_concept, dim_emb)

        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        if separate_qa:
            self.embed_interaction = nn.Embedding(2 * num_concept + 1, dim_emb)
        else:
            self.embed_interaction = nn.Embedding(2, dim_emb)

        self.encoder_layer = EncoderLayer(params, objects)
        encoder_layer_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        dim_model = encoder_layer_config["dim_model"]
        dim_final_fc = encoder_layer_config["dim_final_fc"]
        dropout = encoder_layer_config["dropout"]
        self.predict_layer = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 对性能来说至关重要的一步
        for p in self.parameters():
            if p.size(0) == num_question and num_question > 0:
                torch.nn.init.constant_(p, 0.)

    def get_concept_emb(self):
        return self.embed_concept.weight

    def base_emb(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = self.embed_concept(concept_seq)
        if separate_qa:
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.embed_concept_variation(concept_seq)
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        if separate_qa:
            # uq * f_(ct,rt) + e_(ct,rt)
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)
        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }
        latent = self.encoder_layer(encoder_input)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_latent(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.embed_concept_variation(concept_seq)
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        if separate_qa:
            # uq * f_(ct,rt) + e_(ct,rt)
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)
        encoder_input = {
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }
        latent = self.encoder_layer.get_latent(encoder_input)

        return latent

    def get_latent_last(self, batch):
        latent = self.get_latent(batch)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean(self, batch):
        latent = self.get_latent(batch)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_duo_cl_loss(self, batch, latent_type):
        batch_aug = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_hard_neg = {
            "concept_seq": batch["concept_seq_hard_neg"],
            "question_seq": batch["question_seq_hard_neg"],
            "correct_seq": batch["correct_seq_hard_neg"],
            "mask_seq": batch["mask_seq_hard_neg"]
        }

        if latent_type == "last_time":
            latent_ori_pooled = self.get_latent_last(batch)
            latent_aug_pooled = self.get_latent_last(batch_aug)
            latent_hard_neg_pooled = self.get_latent_last(batch_hard_neg)
        elif latent_type == "mean_pool":
            latent_ori_pooled = self.get_latent_mean(batch)
            latent_aug_pooled = self.get_latent_mean(batch_aug)
            latent_hard_neg_pooled = self.get_latent_mean(batch_hard_neg)
        else:
            raise NotImplementedError()

        temp = self.params["other"]["duo_cl"]["temp"]
        cl_loss = duo_info_nce(latent_ori_pooled, latent_aug_pooled, temp,
                               sim_type="cos", z_hard_neg=latent_hard_neg_pooled)

        return cl_loss

    def get_instance_cl_loss_one_seq(self, batch, latent_type):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        if latent_type == "last_time":
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
        elif latent_type == "mean_pool":
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
        else:
            raise NotImplementedError()

        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim_aug = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug1_pooled.unsqueeze(0),
                                              dim=-1) / temp

        if "correct_seq_hard_neg" in batch.keys():
            batch_hard_neg = {
                "concept_seq": batch["concept_seq"],
                "question_seq": batch["question_seq"],
                "correct_seq": batch["correct_seq_hard_neg"],
                "mask_seq": batch["mask_seq"]
            }
            if latent_type == "last_time":
                latent_hard_neg_pooled = self.get_latent_last(batch_hard_neg)
            elif latent_type == "mean_pool":
                latent_hard_neg_pooled = self.get_latent_mean(batch_hard_neg)
            else:
                raise NotImplementedError()
            cos_sim_neg = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1),
                                                  latent_hard_neg_pooled.unsqueeze(0), dim=-1) / temp
            cos_sim = torch.cat((cos_sim_aug, cos_sim_neg), dim=1)
        else:
            cos_sim = cos_sim_aug

        labels = torch.arange(cos_sim.size(0)).long().to(self.params["device"])
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_instance_cl_loss_all_interaction(self, batch):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        latent_aug0 = self.get_latent(batch_aug0)
        mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)
        latent_aug0_last = latent_aug0[torch.where(mask4last_aug0 == 1)]

        latent_aug1 = self.get_latent(batch_aug1)
        mask4last_aug1 = get_mask4last_or_penultimate(batch_aug1["mask_seq"], penultimate=False)
        latent_aug1_last = latent_aug1[torch.where(mask4last_aug1 == 1)]

        bs = latent_aug0.shape[0]
        seq_len = latent_aug0.shape[1]
        m = (torch.eye(bs) == 0)

        # 将另一增强序列的每个时刻都作为一个neg
        neg_all = latent_aug1.repeat(bs, 1, 1).reshape(bs, bs, seq_len, -1)[m].reshape(bs, bs - 1, seq_len, -1)
        mask_bool4neg = torch.ne(batch["mask_seq_aug_1"].repeat(bs, 1).reshape(bs, bs, -1)[m].reshape(bs, bs - 1, -1),
                                 0)

        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim_list = []
        for i in range(bs):
            anchor = latent_aug0_last[i]
            pos = latent_aug1_last[i]
            neg = neg_all[i][:, 1:][mask_bool4neg[i][:, 1:]]
            sim_i = torch.cosine_similarity(anchor, torch.cat((pos.unsqueeze(dim=0), neg), dim=0)) / temp
            cos_sim_list.append(sim_i.unsqueeze(dim=0))

        labels = torch.tensor([0]).long().to(self.params["device"])
        cl_loss = 0.
        for i in range(bs):
            cos_sim = cos_sim_list[i]
            cl_loss = cl_loss + nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    # def get_instance_cl_loss_one_seq_adv(self, batch, dataset_adv, latent_type):
    #     batch_aug0 = {
    #         "concept_seq": batch["concept_seq_aug_0"],
    #         "question_seq": batch["question_seq_aug_0"],
    #         "correct_seq": batch["correct_seq_aug_0"],
    #         "mask_seq": batch["mask_seq_aug_0"]
    #     }
    #     latent_aug0 = self.get_latent(batch_aug0)[:, 1:]
    #
    #     seq_ids = batch["seq_id"]
    #     emb_seq_aug1 = dataset_adv["emb_seq"][seq_ids.to("cpu")].to(self.params["device"])
    #     encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    #     dim_concept = encoder_config["dim_concept"]
    #     dim_question = encoder_config["dim_question"]
    #     concept_emb = emb_seq_aug1[:, 1:, :dim_concept]
    #     question_emb = emb_seq_aug1[:, 1:, dim_concept:(dim_concept + dim_question)]
    #     _, latent_aug1 = self.forward_from_input_emb(emb_seq_aug1[:, :-1], concept_emb, question_emb)
    #
    #     if latent_type == "last_time":
    #         mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)[:, 1:]
    #         mask4last_aug1 = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
    #         latent_aug0_pooled = latent_aug0[torch.where(mask4last_aug0 == 1)]
    #         latent_aug1_pooled = latent_aug1[torch.where(mask4last_aug1 == 1)]
    #     elif latent_type == "mean_pool":
    #         mask_seq_aug0 = batch["mask_seq_aug_0"][:, 1:]
    #         mask_seq_aug1 = batch["mask_seq"][:, 1:]
    #         latent_aug0_pooled = (latent_aug0 * mask_seq_aug0.unsqueeze(-1)).sum(1) / mask_seq_aug0.sum(-1).unsqueeze(
    #             -1)
    #         latent_aug1_pooled = (latent_aug1 * mask_seq_aug1.unsqueeze(-1)).sum(1) / mask_seq_aug1.sum(-1).unsqueeze(
    #             -1)
    #     else:
    #         raise NotImplementedError()
    #
    #     temp = self.params["other"]["instance_cl"]["temp"]
    #     cos_sim_aug = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug1_pooled.unsqueeze(0),
    #                                           dim=-1) / temp
    #
    #     labels = torch.arange(cos_sim_aug.size(0)).long().to(self.params["device"])
    #     cl_loss = nn.functional.cross_entropy(cos_sim_aug, labels)
    #
    #     return cl_loss

    # def get_instance_cl_loss_all_interaction_adv(self, batch, dataset_adv):
    #     batch_aug0 = {
    #         "concept_seq": batch["concept_seq_aug_0"],
    #         "question_seq": batch["question_seq_aug_0"],
    #         "correct_seq": batch["correct_seq_aug_0"],
    #         "mask_seq": batch["mask_seq_aug_0"]
    #     }
    #     latent_aug0 = self.get_latent(batch_aug0)[:, 1:]
    #     mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)[:, 1:]
    #     latent_aug0_last = latent_aug0[torch.where(mask4last_aug0 == 1)]
    #
    #     seq_ids = batch["seq_id"]
    #     emb_seq_aug1 = dataset_adv["emb_seq"][seq_ids.to("cpu")].to(self.params["device"])
    #     encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    #     dim_concept = encoder_config["dim_concept"]
    #     dim_question = encoder_config["dim_question"]
    #     concept_emb = emb_seq_aug1[:, 1:, :dim_concept]
    #     question_emb = emb_seq_aug1[:, 1:, dim_concept:(dim_concept + dim_question)]
    #     _, latent_aug1 = self.forward_from_input_emb(emb_seq_aug1[:, :-1], concept_emb, question_emb)
    #     mask4last_aug1 = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
    #     latent_aug1_last = latent_aug1[torch.where(mask4last_aug1 == 1)]
    #
    #     bs = latent_aug0.shape[0]
    #     seq_len = latent_aug0.shape[1]
    #     m = (torch.eye(bs) == 0)
    #
    #     # 将另一增强序列的每个时刻都作为一个neg
    #     neg_all = latent_aug1.repeat(bs, 1, 1).reshape(bs, bs, seq_len, -1)[m].reshape(bs, bs - 1, seq_len, -1)
    #     mask_bool4neg = (
    #         torch.ne(batch["mask_seq"][:, 1:].repeat(bs, 1).reshape(bs, bs, -1)[m].reshape(bs, bs - 1, -1), 0))
    #
    #     temp = self.params["other"]["instance_cl"]["temp"]
    #     cos_sim_list = []
    #     for i in range(bs):
    #         anchor = latent_aug0_last[i]
    #         pos = latent_aug1_last[i]
    #         neg = neg_all[i][:, 1:][mask_bool4neg[i][:, 1:]]
    #         sim_i = torch.cosine_similarity(anchor, torch.cat((pos.unsqueeze(dim=0), neg), dim=0)) / temp
    #         cos_sim_list.append(sim_i.unsqueeze(dim=0))
    #
    #     labels = torch.tensor([0]).long().to(self.params["device"])
    #     cl_loss = 0.
    #     for i in range(bs):
    #         cos_sim = cos_sim_list[i]
    #         cl_loss = cl_loss + nn.functional.cross_entropy(cos_sim, labels)
    #
    #     return cl_loss

    def get_cluster_cl_loss_one_seq(self, batch, clus, latent_type):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }
        latent_aug0 = self.get_latent(batch_aug0)
        batch_size = latent_aug0.shape[0]

        if latent_type == "last_time":
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
            latent_ori_pooled = self.get_latent_last(batch)
        elif latent_type == "mean_pool":
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
            latent_ori_pooled = self.get_latent_mean(batch)
        else:
            raise NotImplementedError()
        state = np.array(latent_ori_pooled.detach().cpu().tolist())
        intent_id, intent = clus.query(state)

        intent_id0 = intent_id.contiguous().view(-1, 1)
        intent_id1 = intent_id.contiguous().view(1, -1)
        mask4inf = (intent_id0 == intent_id1) & torch.ne(torch.eye(batch_size), 1).to(self.params["device"])

        temp = self.params["other"]["cluster_cl"]["temp"]
        cos_sim_aug0 = torch.cosine_similarity(intent.unsqueeze(1), latent_aug0_pooled.unsqueeze(0), dim=-1) / temp
        cos_sim_aug1 = torch.cosine_similarity(intent.unsqueeze(1), latent_aug1_pooled.unsqueeze(0), dim=-1) / temp
        cos_sim_aug0[mask4inf] = -(1 / temp)
        cos_sim_aug1[mask4inf] = -(1 / temp)

        labels = torch.arange(batch_size).long().to(self.params["device"])
        cl_loss0 = nn.functional.cross_entropy(cos_sim_aug0, labels)
        cl_loss1 = nn.functional.cross_entropy(cos_sim_aug1, labels)

        return (cl_loss0 + cl_loss1) / 2

    def get_rasch_loss(self, batch):
        question_seq = batch["question_seq"]
        question_difficulty_emb = self.embed_question_difficulty(question_seq)

        return (question_difficulty_emb ** 2.).sum()

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        rasch_loss = self.get_rasch_loss(batch)
        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        return loss

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    # def get_input_emb(self, batch):
    #     pass
    #
    # def forward_from_input_emb(self, input_emb, concept_emb, question_emb):
    #     pass
    #
    # def get_max_entropy_adv_aug_emb(self, batch, adv_learning_rate, loop_adv, eta, gamma):
    #     encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    #     dim_concept = encoder_config["dim_concept"]
    #     dim_question = encoder_config["dim_question"]
    #
    #     correct_seq = batch["correct_seq"]
    #     mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
    #     mask4penultimate = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=True)
    #     ground_truth = correct_seq[torch.where(mask4last == 1)]
    #
    #     latent = self.get_latent(batch)
    #     latent_penultimate = latent[torch.where(mask4penultimate == 1)].detach().clone()
    #
    #     inputs_max = self.get_input_emb(batch).detach().clone()
    #     latent_penultimate.requires_grad_(False)
    #     inputs_max.requires_grad_(True)
    #     optimizer = optim.SGD(params=[inputs_max], lr=adv_learning_rate)
    #     adv_predict_loss = 0.
    #     adv_entropy = 0.
    #     adv_mse_loss = 0.
    #     for ite_max in range(loop_adv):
    #         concept_emb = inputs_max[:, 1:, :dim_concept]
    #         question_emb = inputs_max[:, 1:, dim_concept:(dim_concept + dim_question)]
    #         predict_score, latent = self.forward_from_input_emb(inputs_max[:, :-1], concept_emb, question_emb)
    #         predict_score = predict_score[torch.where(mask4last[:, 1:] == 1)]
    #         adv_pred_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
    #         entropy_loss = binary_entropy(predict_score)
    #         latent_mse_loss = (
    #             nn.functional.mse_loss(latent[torch.where(mask4penultimate[:, :-1] == 1)], latent_penultimate))
    #
    #         if ite_max == (loop_adv - 1):
    #             adv_predict_loss += adv_pred_loss.detach().cpu().item()
    #             adv_entropy += entropy_loss.detach().cpu().item()
    #             adv_mse_loss += latent_mse_loss.detach().cpu().item()
    #         loss = adv_pred_loss + eta * entropy_loss - gamma * latent_mse_loss
    #         self.zero_grad()
    #         optimizer.zero_grad()
    #         (-loss).backward()
    #         optimizer.step()
    #
    #     return inputs_max, adv_predict_loss, adv_entropy, adv_mse_loss
