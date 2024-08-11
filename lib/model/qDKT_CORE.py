import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer


class qDKT_CORE(nn.Module):
    model_name = "qDKT_CORE"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]
        dim_emb = encoder_config["dim_emb"]
        dropout = encoder_config["dropout"]

        self.lstm = nn.LSTM(4 * dim_emb, dim_emb, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.fc = nn.Sequential(
            nn.Linear(3 * dim_emb, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.question_net = nn.Sequential(
            nn.Linear(dim_emb * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # 这里一开始代码写错了，本来是想写user的，写成use了
        # 但是由于后面已经跑了很多实验了，如果改成user，加载模型会报错，所以这里就不改了
        self.use_net = nn.Linear(dim_emb, 2)
        self.softmax = nn.Softmax(-1)

    def get_qc_emb4single_concept(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        concept_question_emb = self.embed_layer.get_emb_concatenated(
            ("concept", "question"), (concept_seq, question_seq)
        )

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], fusion_type="mean")

    def get_predict_score(self, batch):
        # inference和train不一样
        mask_bool_seq = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        _, _, _, logit_Core = self.forward(batch)
        CORE_pred = torch.masked_select(logit_Core, mask_bool_seq)
        predict_score = torch.softmax(CORE_pred.view(-1, 2), dim=-1)[:, 1]

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        _, _, _, logit_Core_DKT = self.forward(batch)
        predict_score = torch.softmax(logit_Core_DKT, dim=-1)[:, :, 1]

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
        loss = predict_loss + KL_loss

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("KL loss", KL_loss.detach().cpu().item() * num_sample, num_sample)

        return loss

    def forward(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]["dim_emb"]

        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)

        # interaction = cat(qc, 0) if correct == 1 else cat(0, qc)
        correct_seq = batch["correct_seq"]
        interaction_emb = torch.cat(
            [
                qc_emb[:, :-1].mul(correct_seq.unsqueeze(-1).repeat(1, 1, 2 * dim_emb)[:, :-1]),
                qc_emb[:, :-1].mul((1 - correct_seq).unsqueeze(-1).repeat(1, 1, 2 * dim_emb)[:, :-1])
            ], dim=-1
        )

        latent, _ = self.lstm(interaction_emb)
        logits = self.fc(torch.cat([latent, qc_emb[:, 1:]], -1))
        q_logits = self.question_net(qc_emb[:, 1:].detach())
        s_logits = self.use_net(latent.detach())

        # z_QKS = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=True, S_fact=True)
        # z_Q = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=False, S_fact=False)
        # z_KS = self.fusion(logits, q_logits, s_logits, q_fact=False, k_fact=True, s_fact=True)
        # z = self.fusion(logits, q_logits, s_logits, q_fact=False, k_fact=False, s_fact=False)

        z_QKS = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=True, S_fact=True)
        z_Q = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=False, S_fact=False)
        logit_Core_DKT = z_QKS - z_Q

        # TIE
        z_nde = self.fusion(logits.clone().detach(), q_logits.clone().detach(), s_logits.clone().detach(),
                            Q_fact=True, K_fact=False, S_fact=False)
        # NDE = z_Q - z
        mask_bool_seq_ = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        z_nde_pred = torch.masked_select(z_nde, mask_bool_seq_).view(-1, 2)
        q_pred = torch.masked_select(q_logits, mask_bool_seq_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_QKS, mask_bool_seq_).view(-1, 2)

        return z_nde_pred, q_pred, z_qks_pred, logit_Core_DKT

    def fusion(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]["fusion_mode"]
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
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT_CORE"]["fusion_mode"]

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
