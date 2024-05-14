import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, num_layer, dim_in, dropout):
        super().__init__()

        self.num_layer = num_layer
        dim_middle = dim_in // 2
        if num_layer == 1:
            self.mlp = nn.Linear(dim_in, 1)
        elif num_layer == 2:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, 1)
            )
        if num_layer > 2:
            self.residual = nn.Linear(dim_in, 1)

    def forward(self, x):
        return self.mlp(x) if (self.num_layer <= 2) else torch.relu(self.mlp(x) + self.residual(x))


class ELMKT(nn.Module):
    model_name = "ELMKT"

    def __init__(self, params, objects):
        super(ELMKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ELMKT"]
        num_question = encoder_config["num_question"]
        dim_emb = encoder_config["dim_emb"]
        dropout = encoder_config["dropout"]
        use_lpkt_predictor = encoder_config["use_lpkt_predictor"]
        num_predictor_layer = encoder_config["num_predictor_layer"]

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_interval_time = nn.Embedding(100, dim_emb)
        self.embed_use_time = nn.Embedding(100, dim_emb)
        self.embed_num_hint = nn.Embedding(100, dim_emb)
        self.embed_num_attempt = nn.Embedding(100, dim_emb)

        self.learning_factor_fuser = nn.Linear(5 * dim_emb, dim_emb)
        self.learning_gain_gate = nn.Linear(4 * dim_emb, dim_emb)
        self.learning_absorb_gate = nn.Linear(4 * dim_emb, dim_emb)
        self.forgetting_gate = nn.Linear(3 * dim_emb, dim_emb)
        if use_lpkt_predictor:
            self.predictor = nn.Linear(2 * dim_emb, dim_emb)
        else:
            self.predictor = Predictor(num_predictor_layer, 2 * dim_emb, dropout)
        self.dropout = nn.Dropout(dropout)
        self.learnable_Q_table = nn.Parameter(self.objects["data"]["Q_table_tensor"].float() * 10, requires_grad=True)

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ELMKT"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        use_learnable_q = encoder_config["use_learnable_q"]
        max_q_value = encoder_config["max_q_unrelated_concept"]
        min_q_value = encoder_config["min_q_related_concept"]
        use_lpkt_predictor = encoder_config["use_lpkt_predictor"]
        device = self.params["device"]

        if use_learnable_q:
            Q_table1 = self.objects["data"]["Q_table_tensor"] + (max_q_value / (1 - min_q_value))
            Q_table1[Q_table1 > 1] = 1
            Q_table2 = (1 - self.objects["data"]["Q_table_tensor"]) * min_q_value
            Q_table3 = self.objects["data"]["Q_table_tensor"] * min_q_value
            q_matrix = (torch.sigmoid(self.learnable_Q_table) * (1 - min_q_value) + Q_table3 - Q_table2) * Q_table1
        else:
            q_matrix = self.objects["data"]["Q_table_tensor"] + 0.03
            q_matrix[q_matrix > 1] = 1

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        batch_size, seq_len = question_seq.size(0), question_seq.size(1)

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.view(-1, 1).repeat(1, dim_emb).view(batch_size, -1, dim_emb)
        if "use_time_seq" in batch.keys():
            use_time_emb = self.embed_use_time(batch["use_time_seq"])
        else:
            use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "interval_time_seq" in batch.keys():
            interval_time_emb = self.embed_use_time(batch["interval_time_seq"])
        else:
            interval_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "num_hint_seq" in batch.keys():
            num_hint_emb = self.embed_use_time(batch["num_hint_seq"])
        else:
            num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "num_attempt_seq" in batch.keys():
            num_attempt_emb = self.embed_use_time(batch["num_attempt_seq"])
        else:
            num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        learning_emb_all_time = self.learning_factor_fuser(
            torch.cat((question_emb, correct_emb, num_attempt_emb, num_hint_emb, use_time_emb), dim=-1)
        )

        hidden_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_emb)).repeat(batch_size, 1, 1).to(device)
        learning_emb_pre = torch.zeros(batch_size, dim_emb).to(device)
        # hidden_tilde：具体习题相关的hidden concept emb聚合
        hidden_tilde_current = None
        # hidden_concept_all_time = torch.zeros(batch_size, seq_len, num_concept, dim_emb).to(device)
        # hidden_all_time = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        predict_score_all_time = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            concept_related_current = q_matrix[question_seq[:, t]].view(batch_size, 1, -1)
            interval_time_emb_current = interval_time_emb[:, t]

            # Learning Module
            if hidden_tilde_current is None:
                hidden_tilde_current = concept_related_current.bmm(hidden_pre).view(batch_size, dim_emb)
            learning_emb_current = learning_emb_all_time[:, t]
            learning_gain = self.learning_gain_gate(
                torch.cat((learning_emb_pre, interval_time_emb_current, learning_emb_current, hidden_tilde_current), dim=-1)
            )
            learning_gain = torch.tanh(learning_gain)
            learning_absorb = self.learning_absorb_gate(
                torch.cat((learning_emb_pre, interval_time_emb_current, learning_emb_current, hidden_tilde_current), dim=-1)
            )
            learning_absorb = torch.sigmoid(learning_absorb) * ((learning_gain + 1) / 2)
            actual_learning_gain = self.dropout(
                concept_related_current.transpose(1, 2).bmm(learning_absorb.view(batch_size, 1, -1))
            )

            # Forgetting Module
            gamma_f = torch.sigmoid(self.forgetting_gate(torch.cat((
                hidden_pre,
                learning_absorb.repeat(1, num_concept).view(batch_size, -1, dim_emb),
                interval_time_emb_current.repeat(1, num_concept).view(batch_size, -1, dim_emb)
            ), 2)))
            hidden_current = actual_learning_gain + gamma_f * hidden_pre

            # Predicting Module
            concept_related_next = q_matrix[question_seq[:, t + 1]].view(batch_size, 1, -1)
            hidden_tilde_next = concept_related_next.bmm(hidden_current).view(batch_size, dim_emb)
            if use_lpkt_predictor:
                predict_score_next = torch.sigmoid(
                    self.predictor(torch.cat((question_emb[:, t + 1], hidden_tilde_next), 1))
                ).sum(dim=-1) / dim_emb
            else:
                predict_score_next = torch.sigmoid(
                    self.predictor(torch.cat((question_emb[:, t + 1], hidden_tilde_next), 1))
                ).squeeze(dim=-1)
            predict_score_all_time[:, t + 1] = predict_score_next

            # prepare for next prediction
            learning_emb_pre = learning_emb_current
            hidden_pre = hidden_current
            hidden_tilde_current = hidden_tilde_next

        return predict_score_all_time

    def get_latent(self, batch, correct_noise_strength=0.):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["ELMKT"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        use_learnable_q = encoder_config["use_learnable_q"]
        max_q_value = encoder_config["max_q_unrelated_concept"]
        min_q_value = encoder_config["min_q_related_concept"]
        device = self.params["device"]

        if use_learnable_q:
            Q_table1 = self.objects["data"]["Q_table_tensor"] + (max_q_value / (1 - min_q_value))
            Q_table1[Q_table1 > 1] = 1
            Q_table2 = (1 - self.objects["data"]["Q_table_tensor"]) * min_q_value
            Q_table3 = self.objects["data"]["Q_table_tensor"] * min_q_value
            q_matrix = (torch.sigmoid(self.learnable_Q_table) * (1 - min_q_value) + Q_table3 - Q_table2) * Q_table1
        else:
            q_matrix = self.objects["data"]["Q_table_tensor"] + 0.03
            q_matrix[q_matrix > 1] = 1

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        batch_size, seq_len = question_seq.size(0), question_seq.size(1)

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.view(-1, 1).repeat(1, dim_emb).view(batch_size, -1, dim_emb)
        if 0 < correct_noise_strength < 0.5:
            noise = torch.rand(batch_size, seq_len, dim_emb).to(self.params["device"]) * correct_noise_strength
            noise_mask = torch.ones_like(noise).float().to(self.params["device"])
            noise_mask[correct_seq == 1] = -1
            noise = noise * noise_mask
            correct_emb = correct_emb + noise
        if "use_time_seq" in batch.keys():
            use_time_emb = self.embed_use_time(batch["use_time_seq"])
        else:
            use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "interval_time_seq" in batch.keys():
            interval_time_emb = self.embed_use_time(batch["interval_time_seq"])
        else:
            interval_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "num_hint_seq" in batch.keys():
            num_hint_emb = self.embed_use_time(batch["num_hint_seq"])
        else:
            num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        if "num_attempt_seq" in batch.keys():
            num_attempt_emb = self.embed_use_time(batch["num_attempt_seq"])
        else:
            num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(device)
        learning_emb_all_time = self.learning_factor_fuser(
            torch.cat((question_emb, correct_emb, num_attempt_emb, num_hint_emb, use_time_emb), dim=-1)
        )

        hidden_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_emb)).repeat(batch_size, 1, 1).to(device)
        learning_emb_pre = torch.zeros(batch_size, dim_emb).to(device)
        # hidden_tilde：具体习题相关的hidden concept emb聚合
        hidden_tilde_current = None
        hidden_tilde_all_time = torch.zeros(batch_size, seq_len, dim_emb).to(device)

        for t in range(0, seq_len - 1):
            concept_related_current = q_matrix[question_seq[:, t]].view(batch_size, 1, -1)
            interval_time_emb_current = interval_time_emb[:, t]

            # Learning Module
            if hidden_tilde_current is None:
                hidden_tilde_current = concept_related_current.bmm(hidden_pre).view(batch_size, dim_emb)
            learning_emb_current = learning_emb_all_time[:, t]
            learning_gain = self.learning_gain_gate(
                torch.cat((learning_emb_pre, interval_time_emb_current, learning_emb_current, hidden_tilde_current),
                          dim=-1)
            )
            learning_gain = torch.tanh(learning_gain)
            learning_absorb = self.learning_absorb_gate(
                torch.cat((learning_emb_pre, interval_time_emb_current, learning_emb_current, hidden_tilde_current),
                          dim=-1)
            )
            learning_absorb = torch.sigmoid(learning_absorb) * ((learning_gain + 1) / 2)
            actual_learning_gain = self.dropout(
                concept_related_current.transpose(1, 2).bmm(learning_absorb.view(batch_size, 1, -1))
            )

            # Forgetting Module
            gamma_f = torch.sigmoid(self.forgetting_gate(torch.cat((
                hidden_pre,
                learning_absorb.repeat(1, num_concept).view(batch_size, -1, dim_emb),
                interval_time_emb_current.repeat(1, num_concept).view(batch_size, -1, dim_emb)
            ), 2)))
            hidden_current = actual_learning_gain + gamma_f * hidden_pre

            # Predicting Module
            concept_related_next = q_matrix[question_seq[:, t + 1]].view(batch_size, 1, -1)
            hidden_tilde_next = concept_related_next.bmm(hidden_current).view(batch_size, dim_emb)
            hidden_tilde_all_time[:, t + 1] = hidden_tilde_next

        return hidden_tilde_all_time

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]

    def get_predict_loss(self, batch, loss_record=None):
        w_cl_loss = self.params["loss_config"].get("cl loss", 0)
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        batch_size = mask_bool_seq.shape[0]

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        unbias_loss = self.get_unbias_loss(batch)
        if loss_record is not None:
            loss_record.add_loss("cl loss", unbias_loss.detach().cpu().item() * batch_size, batch_size)
        loss = predict_loss + unbias_loss * w_cl_loss

        return loss

    def get_unbias_loss(self, batch):
        temp = self.params["other"]["instance_cl"]["temp"]
        correct_noise = self.params["other"]["instance_cl"]["correct_noise"]

        batch_size = batch["mask_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])

        latent_aug0 = self.get_latent(batch, correct_noise_strength=correct_noise)
        latent_aug0 = latent_aug0[first_index, batch["seq_len"] - 1]
        latent_aug1 = self.get_latent(batch, correct_noise_strength=correct_noise)
        latent_aug1 = latent_aug1[first_index, batch["seq_len"] - 1]

        cos_sim = torch.cosine_similarity(latent_aug0.unsqueeze(1), latent_aug1.unsqueeze(0), dim=-1) / temp
        batch_size = cos_sim.size(0)
        labels = torch.arange(batch_size).long().to(self.params["device"])
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss
