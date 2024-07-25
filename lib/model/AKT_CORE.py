import torch
from torch import nn

from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.AKT_Block import Architecture


class AKT_CORE(nn.Module):
    model_name = "AKT_CORE"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        num_concept = encoder_config["num_concept"]
        num_question = encoder_config["num_question"]
        dim_model = encoder_config["dim_model"]
        dropout = encoder_config["dropout"]
        dim_final_fc = encoder_config["dim_final_fc"]
        separate_qa = encoder_config["separate_qa"]

        self.question_net = nn.Sequential(
            nn.Linear(dim_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

        self.user_net = nn.Linear(dim_model, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.fusion_mode = 'sum'
        self.constant = nn.Parameter(torch.tensor(0.0))

        self.embed_question_difficulty = nn.Embedding(num_question, 1)
        self.embed_concept_variation = nn.Embedding(num_concept, dim_model, padding_idx=0)
        self.embed_interaction_variation = nn.Embedding(2 * num_concept, dim_model)

        self.embed_concept = nn.Embedding(num_concept, dim_model, padding_idx=0)
        if separate_qa:
            self.embed_interaction = nn.Embedding(2 * num_concept, dim_model)
        else:
            self.embed_interaction = nn.Embedding(2, dim_model)

        self.model = Architecture(params)

        self.out = nn.Sequential(
            nn.Linear(dim_model + dim_model, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

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
        separate_qa = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]["separate_qa"]

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
            interaction_emb = interaction_emb + question_difficulty_emb * (
                        interaction_variation_emb + concept_variation_emb)

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }
        d_output, s_output = self.model(encoder_input, is_core=True)

        concat_q = torch.cat([d_output, question_emb], dim=-1)
        output = self.out(concat_q)
        q_logit = self.question_net(question_emb.detach())[:, 1:]
        s_logit = self.user_net(s_output.detach())[:, :-1]
        logits = output[:, 1:]
        z_qks = self.fusion(logits, q_logit, s_logit, Q_fact=True, K_fact=True, S_fact=True)
        z_q = self.fusion(logits, q_logit, s_logit, Q_fact=True, K_fact=False, S_fact=False)
        logit_Core_AKT = z_qks - z_q

        z_nde = self.fusion(logits.clone().detach(), q_logit.clone().detach(), s_logit.clone().detach(),
                            Q_fact=True, K_fact=False, S_fact=False)
        # NDE = z_Q - z
        mask_bool_seq_ = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        z_nde_pred = torch.masked_select(z_nde, mask_bool_seq_).view(-1, 2)
        q_pred = torch.masked_select(q_logit, mask_bool_seq_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_qks, mask_bool_seq_).view(-1, 2)

        return z_nde_pred, q_pred, z_qks_pred, logit_Core_AKT

    def get_predict_score(self, batch):
        # inference和train不一样
        mask_bool_seq = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        _, _, _, logit_Core = self.forward(batch)
        CORE_pred = torch.masked_select(logit_Core, mask_bool_seq)
        predict_score = torch.softmax(CORE_pred.view(-1, 2), dim=-1)[:, 1]

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        _, _, _, logit_Core = self.forward(batch)
        predict_score = torch.softmax(logit_Core, dim=-1)[:, :, 1]

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        z_nde_pred, q_pred, z_qks_pred, CORE_pred = self.forward(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = torch.nn.functional.cross_entropy(z_qks_pred, ground_truth) + \
                       torch.nn.functional.cross_entropy(q_pred, ground_truth)
        p_te = self.softmax(z_qks_pred).clone().detach()
        KL_loss = - p_te * self.softmax(z_nde_pred).log()
        KL_loss = KL_loss.sum(1).mean()

        question_seq = batch["question_seq"]
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        rasch_loss = (question_difficulty_emb ** 2.).sum()

        loss = predict_loss + KL_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("KL loss", KL_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("rasch_loss", rasch_loss.detach().cpu().item(), 1)

        return loss

    def fusion(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]["fusion_mode"]
        predict_K, predict_Q, predict_S = self.transform(predict_K, predict_Q, predict_S, Q_fact, K_fact, S_fact)

        if fusion_mode == 'rubin':
            # 鲁宾因果模型（潜在因果框架）
            z = predict_K * torch.sigmoid(predict_Q)

        elif fusion_mode == 'hm':
            #
            z = predict_K * predict_S * predict_Q
            z = torch.log(z + 1e-12) - torch.log1p(z)

        elif fusion_mode == 'sum':
            z = predict_K + predict_Q + predict_S
            z = torch.log(torch.sigmoid(z) + 1e-12)

        else:
            raise NotImplementedError()

        return z

    def transform(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]["fusion_mode"]

        if not K_fact:
            predict_K = self.constant * torch.ones_like(predict_K).to(self.params["device"])

        if not Q_fact:
            predict_Q = self.constant * torch.ones_like(predict_Q).to(self.params["device"])

        if not S_fact:
            predict_S = self.constant * torch.ones_like(predict_S).to(self.params["device"])

        if fusion_mode == 'hm':
            predict_K = torch.sigmoid(predict_K)
            predict_Q = torch.sigmoid(predict_Q)
            predict_S = torch.sigmoid(predict_S)

        return predict_K, predict_Q, predict_S
