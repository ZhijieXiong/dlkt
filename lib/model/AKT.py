import torch
import torch.nn as nn

from .BaseModel4CL import BaseModel4CL
from .Module.EncoderLayer import EncoderLayer
from .loss_util import binary_entropy
from .util import get_mask4last_or_penultimate


class AKT(nn.Module, BaseModel4CL):
    model_name = "AKT"

    def __init__(self, params, objects):
        # MRO：这种继承情况下AKT的MRO是[AKT, nn.Module, AKT]，即查找方法时的顺序
        # 调用super(AKT, self).__init__()实际上是调用的AKT的“MRO父类”的init方法，即nn.Module
        # 调用super(nn.Module, self).__init__()实际上是调用的nn.Module的“MRO父类”的init方法，即BaseModel4CL
        super(AKT, self).__init__()
        super(nn.Module, self).__init__(params, objects)

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
        seq_representation = encoder_config.get("seq_representation", "encoder_output")
        if seq_representation == "knowledge_encoder_output":
            latent = self.encoder_layer.get_latent(encoder_input)
        else:
            latent = self.encoder_layer(encoder_input)

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

    def get_latent_from_adv_data(self, dataset, batch):
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
        if encoder_config["seq_representation"] == "knowledge_encoder_output":
            latent = self.encoder_layer.get_latent(encoder_input)
        else:
            latent = self.encoder_layer(encoder_input)

        return latent

    def get_latent_last_from_adv_data(self, dataset, batch):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch):
        latent = self.get_latent_from_adv_data(dataset, batch)
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

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)

        return predict_score[mask4last.bool()]
