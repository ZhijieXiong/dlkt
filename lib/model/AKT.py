import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from .BaseModel4CL import BaseModel4CL
from .Module.EncoderLayer import EncoderLayer
from .Module.KTEmbedLayer import KTEmbedLayer
from .loss_util import binary_entropy
from .util import get_mask4last_or_penultimate, parse_question_zero_shot


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

        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dropout = encoder_config["dropout"]
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

        # 解析q table
        self.question_head4zero = None
        self.embed_question_difficulty4zero = None
        self.embed_question4zero = None
        self.embed_interaction4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.objects["data"]["question2concept"],
                                                               self.objects["data"]["concept2question"])

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
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = self.get_concept_emb(batch)
        if separate_qa:
            # todo: 有问题，如果是only question也要融合interaction_emb
            concept_seq = batch["concept_seq"]
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
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

    def get_latent(self, batch, use_emb_dropout=False, dropout=0.1):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
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
        if use_emb_dropout:
            question_emb = torch.dropout(question_emb, dropout, self.training)
            interaction_emb = torch.dropout(interaction_emb, dropout, self.training)
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

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def set_emb4zero(self):
        """
        transfer head to tail use gaussian distribution
        :return:
        """
        # 只对question_difficulty_emb进行迁移是没效果的
        data_type = self.params["datasets_config"]["data_type"]
        indices = []
        tail_qs_emb = []
        for z_q, head_qs in self.question_head4zero.items():
            indices.append(z_q)
            head_qs_emb = self.embed_question_difficulty(
                torch.tensor(head_qs).long().to(self.params["device"])
            )
            # len(head_qs_emb)值越小越好，即直接取均值不好，可能fit时报错: pvals < 0, pvals > 1 or pvals contains NaNs
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

            # 取平均没用
            # if len(head_qs_emb) == 0:
            #     tail_q_emb = self.embed_question_difficulty.weight.mean().detach().clone()
            # else:
            #     tail_q_emb = head_qs_emb.mean().detach().clone()
            tail_qs_emb.append(tail_q_emb)
        indices = torch.tensor(indices)
        tail_qs_emb = torch.tensor(tail_qs_emb)
        embed_question_difficulty = self.embed_question_difficulty.weight.detach().clone()
        embed_question_difficulty[indices, 0] = tail_qs_emb
        self.embed_question_difficulty4zero = nn.Embedding(
            embed_question_difficulty.shape[0],
            embed_question_difficulty.shape[1],
            _weight=embed_question_difficulty
        )

    def get_predict_score4question_zero(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
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

    def get_predict_enhance_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        enhance_method = self.params["other"]["output_enhance"]["enhance_method"]
        weight_enhance_loss1 = self.params["loss_config"]["enhance loss 1"]
        weight_enhance_loss2 = self.params["loss_config"]["enhance loss 2"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        if separate_qa:
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)
        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        loss = 0.
        # 预测损失
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        latent = self.encoder_layer(encoder_input)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score_all = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score_all[:, 1:], mask_bool_seq[:, 1:])

        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        rasch_loss = self.get_rasch_loss(batch)

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("rasch_loss", rasch_loss.detach().cpu().item(), 1)
        loss = loss + predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        # enhance method 1: 对于easier和harder习题的损失
        if enhance_method == 0 or enhance_method == 1:
            mask_bool_seq_easier = torch.ne(batch["mask_easier_seq"], 0)
            mask_bool_seq_harder = torch.ne(batch["mask_harder_seq"], 0)
            weight_easier = torch.masked_select(batch["weight_easier_seq"][:, 1:], mask_bool_seq_easier[:, 1:])
            weight_harder = torch.masked_select(batch["weight_harder_seq"][:, 1:], mask_bool_seq_harder[:, 1:])

            concept_emb_easier = self.embed_concept(batch["concept_easier_seq"])
            concept_variation_emb_easier = self.embed_concept_variation(batch["concept_easier_seq"])
            question_difficulty_emb_easier = self.embed_question_difficulty(batch["question_easier_seq"])
            question_emb_easier = concept_emb_easier + question_difficulty_emb_easier * concept_variation_emb_easier

            concept_emb_harder = self.embed_concept(batch["concept_harder_seq"])
            concept_variation_emb_harder = self.embed_concept_variation(batch["concept_harder_seq"])
            question_difficulty_emb_harder = self.embed_question_difficulty(batch["question_harder_seq"])
            question_emb_harder = concept_emb_harder + question_difficulty_emb_harder * concept_variation_emb_harder

            predict_input_easier = torch.cat((latent, question_emb_easier), dim=2)
            predict_score_easier_all = self.predict_layer(predict_input_easier).squeeze(dim=-1)
            predict_input_harder = torch.cat((latent, question_emb_harder), dim=2)
            predict_score_harder_all = self.predict_layer(predict_input_harder).squeeze(dim=-1)

            predict_score_diff1 = predict_score_easier_all[:, 1:] - predict_score_all[:, 1:]
            predict_score_diff1 = torch.masked_select(predict_score_diff1, mask_bool_seq_easier[:, 1:])
            predict_score_diff2 = predict_score_all[:, 1:] - predict_score_harder_all[:, 1:]
            predict_score_diff2 = torch.masked_select(predict_score_diff2, mask_bool_seq_harder[:, 1:])
            enhance_loss_easier = -torch.min(torch.zeros_like(predict_score_diff1).to(self.params["device"]), predict_score_diff1)
            enhance_loss_easier = enhance_loss_easier * weight_easier
            enhance_loss_harder = -torch.min(torch.zeros_like(predict_score_diff2).to(self.params["device"]), predict_score_diff2)
            enhance_loss_harder = enhance_loss_harder * weight_harder

            enhance_loss1 = enhance_loss_easier.mean() + enhance_loss_harder.mean()

            if loss_record is not None:
                loss_record.add_loss("enhance loss 1", enhance_loss1.detach().cpu().item(), 1)
            loss = loss + enhance_loss1 * weight_enhance_loss1

        # enhance loss2: 对于zero shot的习题，用单调理论约束
        if enhance_method == 0 or enhance_method == 2:
            mask_zero_shot_seq = torch.ne(batch["mask_zero_shot_seq"], 0)
            latent_current = latent[:, :-1]
            latent_next = latent[:, 1:]

            concept_emb4zero = self.embed_concept(batch["concept_zero_shot_seq"])
            concept_variation_emb4zero = self.embed_concept_variation(batch["concept_zero_shot_seq"])
            question_difficulty_emb4zero = self.embed_question_difficulty(batch["question_zero_shot_seq"])
            question_emb4zero = concept_emb4zero + question_difficulty_emb4zero * concept_variation_emb4zero

            predict_input_current4zero = torch.cat((latent_current, question_emb4zero[:, :-1]), dim=2)
            predict_input_next4zero = torch.cat((latent_next, question_emb4zero[:, :-1]), dim=2)
            predict_score_current4zero = self.predict_layer(predict_input_current4zero).squeeze(dim=-1)
            predict_score_next4zero = self.predict_layer(predict_input_next4zero).squeeze(dim=-1)
            predict_score_diff4zero = torch.masked_select(predict_score_next4zero - predict_score_current4zero,
                                                          mask_zero_shot_seq[:, :-1])
            enhance_loss2 = -torch.min(torch.zeros_like(predict_score_diff4zero).to(self.params["device"]), predict_score_diff4zero).mean()

            if loss_record is not None:
                loss_record.add_loss("enhance loss 2", enhance_loss2.detach().cpu().item(), 1)
            loss = loss + enhance_loss2 * weight_enhance_loss2

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
        data_type = self.params["datasets_config"]["data_type"]
        correct_seq = batch["correct_seq"]

        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                dataset["embed_concept"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = dataset["embed_concept"](batch["concept_seq"])
        if separate_qa:
            # todo: 有问题，如果是only question也要融合
            concept_seq = batch["concept_seq"]
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = dataset["embed_interaction"](interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = dataset["embed_interaction"](correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        data_type = self.params["datasets_config"]["data_type"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        concept_emb, interaction_emb = self.base_emb_from_adv_data(dataset, batch)
        if data_type == "only_question":
            concept_variation_emb = KTEmbedLayer.concept_fused_emb(
                dataset["embed_concept_variation"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_variation_emb = dataset["embed_concept_variation"](batch["concept_seq"])
        question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = dataset["embed_interaction_variation"](correct_seq)
        if separate_qa:
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
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
        data_type = self.params["datasets_config"]["data_type"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        concept_emb, interaction_emb = self.base_emb_from_adv_data(dataset, batch)
        if data_type == "only_question":
            concept_variation_emb = KTEmbedLayer.concept_fused_emb(
                dataset["embed_concept_variation"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_variation_emb = dataset["embed_concept_variation"](batch["concept_seq"])
        question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = dataset["embed_interaction_variation"](correct_seq)
        if separate_qa:
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
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

    def get_latent_last_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
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
            # 防止梯度爆炸
            nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=10)
            optimizer.step()

        return adv_predict_loss, adv_entropy, adv_mse_loss

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)

        return predict_score[mask4last.bool()]
