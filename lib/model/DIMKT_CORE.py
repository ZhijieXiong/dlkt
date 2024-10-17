from .Module.KTEmbedLayer import KTEmbedLayer
from .util import *


class DIMKT_CORE(nn.Module):
    model_name = "DIMKT_CORE"
    use_question = True

    def __init__(self, params, objects):
        super(DIMKT_CORE, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        num_question_diff = encoder_config["num_question_diff"]
        num_concept_diff = encoder_config["num_concept_diff"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_question_diff = nn.Embedding(num_question_diff, dim_emb)
        self.embed_concept_diff = nn.Embedding(num_concept_diff, dim_emb)
        self.embed_correct = nn.Embedding(2, dim_emb)

        self.generate_x_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.SDF_MLP1 = nn.Linear(dim_emb, dim_emb)
        self.SDF_MLP2 = nn.Linear(dim_emb, dim_emb)
        self.PKA_MLP1 = nn.Linear(2 * dim_emb, dim_emb)
        self.PKA_MLP2 = nn.Linear(2 * dim_emb, dim_emb)
        self.knowledge_indicator_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.predict_layer = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, 2)
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.question_net = nn.Sequential(
            nn.Linear(dim_emb * 4, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, 2)
        )
        self.user_net = nn.Linear(dim_emb, 2)
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.softmax = nn.Softmax(-1)

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

    def get_concept_diff_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            diff_fuse_table = self.objects["dimkt"]["diff_fuse_table"]
            concept_diff_emb = KTEmbedLayer.other_fused_emb(
                self.embed_concept_diff,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                diff_fuse_table,
                fusion_type="mean"
            )
        else:
            concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])

        return concept_diff_emb

    def forward(self, batch):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_question(batch["question_seq"])
        concept_emb = self.get_concept_emb(batch)
        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        concept_diff_emb = self.get_concept_diff_emb(batch)
        correct_emb = self.embed_correct(batch["correct_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len, 2).to(self.params["device"])

        for t in range(seq_len-1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
                question_emb[:, t+1],
                concept_emb[:, t+1],
                question_diff_emb[:, t+1],
                concept_diff_emb[:, t+1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = self.predict_layer(torch.cat([x_next, h], dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        logits = y[:, :-1]
        qc_emb = torch.cat((question_emb, concept_emb, question_diff_emb, concept_diff_emb), dim=-1)
        q_logits = self.question_net(qc_emb.detach())[:, 1:]
        s_logits = self.user_net(latent.detach())[:, 1:]

        # z_QKS = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=True, S_fact=True)
        # z_Q = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=False, S_fact=False)
        # z_KS = self.fusion(logits, q_logits, s_logits, q_fact=False, k_fact=True, s_fact=True)
        # z = self.fusion(logits, q_logits, s_logits, q_fact=False, k_fact=False, s_fact=False)

        z_QKS = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=True, S_fact=True)
        z_Q = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=False, S_fact=False)
        logit_Core = z_QKS - z_Q

        # TIE
        z_nde = self.fusion(logits.clone().detach(), q_logits.clone().detach(), s_logits.clone().detach(),
                            Q_fact=True, K_fact=False, S_fact=False)
        # NDE = z_Q - z
        mask_bool_seq_ = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        z_nde_pred = torch.masked_select(z_nde, mask_bool_seq_).view(-1, 2)
        q_pred = torch.masked_select(q_logits, mask_bool_seq_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_QKS, mask_bool_seq_).view(-1, 2)

        return z_nde_pred, q_pred, z_qks_pred, logit_Core

    def get_predict_score(self, batch):
        # inference和train不一样
        mask_bool_seq = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        _, _, _, logit_Core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_Core, dim=-1)[:, :, 1]
        CORE_pred = torch.masked_select(logit_Core, mask_bool_seq)
        predict_score = torch.softmax(CORE_pred.view(-1, 2), dim=-1)[:, 1]

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score_seq_len_minus1(self, batch):
        _, _, _, logit_Core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_Core, dim=-1)[:, :, 1]

        return predict_score_batch

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        z_nde_pred, q_pred, z_qks_pred, logit_Core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_Core, dim=-1)[:, :, 1]
        CORE_pred = torch.masked_select(logit_Core, mask_bool_seq[:, 1:].unsqueeze(-1))
        predict_score = torch.softmax(CORE_pred.view(-1, 2), dim=-1)[:, 1]

        predict_loss = (torch.nn.functional.cross_entropy(z_qks_pred, ground_truth) +
                        torch.nn.functional.cross_entropy(q_pred, ground_truth))
        p_te = self.softmax(z_qks_pred).clone().detach()
        KL_loss = - p_te * self.softmax(z_nde_pred).log()
        KL_loss = KL_loss.sum(1).mean()
        loss = predict_loss + KL_loss

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "KL loss": {
                    "value": KL_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def fusion(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["fusion_mode"]
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
        fusion_mode = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["fusion_mode"]

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
