import torch
import torch.nn as nn
import numpy as np

from .Module.EncoderLayer import EncoderLayer
from .loss_util import duo_info_nce, binary_entropy, meta_contrast_rl
from .util import get_mask4last_or_penultimate


class AKT(nn.Module):
    model_name = "AKT"

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

    def get_instance_cl_loss_one_seq(self, batch, latent_type, use_random_seq_len=False):
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
            if use_random_seq_len:
                batch_hard_neg = {
                    "concept_seq": batch["concept_seq_random_len"],
                    "question_seq": batch["question_seq_random_len"],
                    "correct_seq": batch["correct_seq_random_len"],
                    "mask_seq": batch["mask_seq_random_len"]
                }
            else:
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

    def get_cluster_cl_loss_one_seq(self, batch, clus, latent_type, use_random_seq_len=False):
        if use_random_seq_len:
            batch_ori = {
                "concept_seq": batch["concept_seq_random_len"],
                "question_seq": batch["question_seq_random_len"],
                "correct_seq": batch["correct_seq_random_len"],
                "mask_seq": batch["mask_seq_random_len"]
            }
        else:
            batch_ori = batch
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
        batch_size = batch["mask_seq"].shape[0]

        if latent_type == "last_time":
            latent_aug0_pooled = self.get_latent_last(batch_aug0)
            latent_aug1_pooled = self.get_latent_last(batch_aug1)
            latent_ori_pooled = self.get_latent_last(batch_ori)
        elif latent_type == "mean_pool":
            latent_aug0_pooled = self.get_latent_mean(batch_aug0)
            latent_aug1_pooled = self.get_latent_mean(batch_aug1)
            latent_ori_pooled = self.get_latent_mean(batch_ori)
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

    def meta_contrast(self, batch, latent_type, meta_extractors, use_regularization):
        batch_size = batch["mask_seq"].shape[0]
        extractor0, extractor1 = meta_extractors

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
        latent_aug0_extracted = extractor0(latent_aug0_pooled)
        latent_aug1_extracted = extractor1(latent_aug1_pooled)

        temp = self.params["other"]["instance_cl"]["temp"]
        labels = torch.arange(batch_size).long().to(self.params["device"])
        # 随机增强的对比损失
        cos_sim_0 = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug1_pooled.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_0 = nn.functional.cross_entropy(cos_sim_0, labels)

        # 计算meta cl loss
        cos_sim_1 = torch.cosine_similarity(latent_aug0_pooled.unsqueeze(1), latent_aug0_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_1 = nn.functional.cross_entropy(cos_sim_1, labels)

        cos_sim_2 = torch.cosine_similarity(latent_aug1_pooled.unsqueeze(1), latent_aug1_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_2 = nn.functional.cross_entropy(cos_sim_2, labels)

        cos_sim_3 = torch.cosine_similarity(latent_aug0_extracted.unsqueeze(1), latent_aug1_extracted.unsqueeze(0),
                                            dim=-1) / temp
        cl_loss_3 = nn.functional.cross_entropy(cos_sim_3, labels)

        if use_regularization:
            rl_loss = 0.
            rl_loss += meta_contrast_rl(latent_aug0_pooled, latent_aug1_extracted, temp, "dot")
            rl_loss += meta_contrast_rl(latent_aug1_pooled, latent_aug0_extracted, temp, "dot")
            rl_loss += meta_contrast_rl(latent_aug0_extracted, latent_aug1_extracted, temp, "dot")
        else:
            rl_loss = None

        return cl_loss_0, cl_loss_1 + cl_loss_2 + cl_loss_3, rl_loss

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

    def get_latent(self, batch, latent_cl4kt=False):
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
        if latent_cl4kt:
            latent = self.encoder_layer.get_latent(encoder_input)
        else:
            latent = self.encoder_layer(encoder_input)

        return latent

    def get_latent_last(self, batch, latent_cl4kt=False):
        latent = self.get_latent(batch, latent_cl4kt)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean(self, batch, latent_cl4kt=False):
        latent = self.get_latent(batch, latent_cl4kt)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        rasch_loss = self.get_rasch_loss(batch)

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("rasch_loss", rasch_loss.detach().cpu().item(), 1)

        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        return loss

    def get_rasch_loss(self, batch):
        question_seq = batch["question_seq"]
        question_difficulty_emb = self.embed_question_difficulty(question_seq)

        return (question_difficulty_emb ** 2.).sum()

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]

    def base_emb_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = dataset["embed_concept"](concept_seq)
        if separate_qa:
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = dataset["embed_interaction"](interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = dataset["embed_interaction"](correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb_from_adv_data(dataset, batch)
        concept_variation_emb = dataset["embed_concept_variation"](concept_seq)
        question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = dataset["embed_interaction_variation"](correct_seq)
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

    def get_latent_from_adv_data(self, dataset, batch, latent_cl4kt=False):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb_from_adv_data(dataset, batch)
        concept_variation_emb = dataset["embed_concept_variation"](concept_seq)
        question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = dataset["embed_interaction_variation"](correct_seq)
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
        if latent_cl4kt:
            latent = self.encoder_layer.get_latent(encoder_input)
        else:
            latent = self.encoder_layer(encoder_input)

        return latent

    def get_latent_last_from_adv_data(self, dataset, batch, latent_cl4kt=False):
        latent = self.get_latent_from_adv_data(dataset, batch, latent_cl4kt)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch, latent_cl4kt=False):
        latent = self.get_latent_from_adv_data(dataset, batch, latent_cl4kt)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_predict_score_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward_from_adv_data(dataset, batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score_from_adv_data(dataset, batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        rasch_loss = self.get_rasch_loss(batch)
        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        return loss

    def max_entropy_adv_aug(self, dataset, batch, optimizer, loop_adv, eta, gamma):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        latent_ori = self.get_latent_from_adv_data(dataset, batch).detach().clone()
        latent_ori = latent_ori[mask_bool_seq]
        latent_ori.requires_grad_(False)
        adv_predict_loss = 0.
        adv_entropy = 0.
        adv_mse_loss = 0.
        for ite_max in range(loop_adv):
            predict_score = self.get_predict_score_from_adv_data(dataset, batch)
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
            question_seq = batch["question_seq"]
            question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
            rasch_loss = (question_difficulty_emb ** 2.).sum()
            predict_loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]
            entropy_loss = binary_entropy(predict_score)
            latent = self.get_latent_from_adv_data(dataset, batch)
            latent = latent[mask_bool_seq]
            latent_mse_loss = nn.functional.mse_loss(latent, latent_ori)

            if ite_max == (loop_adv - 1):
                adv_predict_loss += predict_loss.detach().cpu().item()
                adv_entropy += entropy_loss.detach().cpu().item()
                adv_mse_loss += latent_mse_loss.detach().cpu().item()
            loss = predict_loss + eta * entropy_loss - gamma * latent_mse_loss
            self.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

        return adv_predict_loss, adv_entropy, adv_mse_loss
