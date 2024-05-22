from .util import *
from .AuxInfoDCT import MLP4Proj


class LBMKT(nn.Module):
    model_name = "LBMKT"

    def __init__(self, params, objects):
        super(LBMKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LBMKT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        num_proj_layer = encoder_config["num_proj_layer"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question + 1, dim_emb)
        self.embed_concept = nn.Embedding(num_concept + 1, dim_emb)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        torch.nn.init.xavier_uniform_(self.embed_concept.weight)
        self.embed_interval_time = nn.Embedding(100, dim_emb)
        self.embed_use_time = nn.Embedding(100, dim_emb)
        self.embed_num_hint = nn.Embedding(100, dim_emb)
        self.embed_num_attempt = nn.Embedding(100, dim_emb)

        self.aux_emb_fusion = nn.Linear(dim_emb * 3, dim_emb)
        self.interaction_fusion = nn.Linear(dim_emb * 4, dim_emb)
        self.gate4know_absorb = nn.Linear(dim_emb * 2, dim_emb)
        self.gate4forget = nn.Linear(dim_emb * 3, dim_emb)
        self.learning_gain_trans = nn.Linear(dim_emb * 3, dim_emb)

        self.que2difficulty = MLP4Proj(num_proj_layer, dim_emb, num_concept, dropout)
        self.latent2ability = MLP4Proj(num_proj_layer, dim_emb, num_concept, dropout)
        self.que2discrimination = MLP4Proj(num_proj_layer, dim_emb, 1, dropout)
        self.dropout = nn.Dropout(dropout)

    def get_concept_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "single_concept":
            concept_emb = self.embed_concept(batch["concept_seq"])
        else:
            q2c_table = self.objects["data"]["q2c_table"]
            q2c_mask_table = self.objects["data"]["q2c_mask_table"]
            question_seq = batch["question_seq"]
            question_emb = self.embed_question(question_seq)
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
            qc_relate = torch.gather(que_difficulty, 2, q2c_table[question_seq]) * q2c_mask_table[question_seq]
            sum_qc_relate = torch.sum(qc_relate, dim=-1, keepdim=True) + 1e-6
            concept_emb_relate = qc_relate.unsqueeze(-1) * self.embed_concept(q2c_table[question_seq])
            concept_emb = torch.sum(concept_emb_relate, dim=-2) / sum_qc_relate

        return concept_emb

    def calculate_predict_score(self, latent, question_emb, question_seq):
        max_que_disc = self.params["models_config"]["kt_model"]["encoder_layer"]["LBMKT"]["max_que_disc"]
        Q_table = self.objects["data"]["Q_table_tensor"]

        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))) * max_que_disc
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        user_ability_ = user_ability * Q_table[question_seq]
        que_difficulty_ = que_difficulty * Q_table[question_seq]
        # 使用补偿性模型，即对于多知识点习题，在考察的一个知识点上的不足可以由其它知识点补偿，同时考虑习题和知识点的关联强度
        sum_weight_concept = torch.sum(que_difficulty_, dim=-1, keepdim=True) + 1e-6
        irt_logits = que_discrimination * (user_ability_ - que_difficulty_) / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(irt_logits, dim=-1))

        return predict_score

    def get_empty_emb(self, shapes):
        return torch.zeros(*shapes).to(self.params["device"])

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LBMKT"]
        dim_emb = encoder_config["dim_emb"]

        has_use_time = "use_time_seq" in batch.keys()
        has_num_hint = "num_hint_seq" in batch.keys()
        has_num_attempt = "num_attempt_seq" in batch.keys()
        has_time = "interval_time_seq" in batch.keys()

        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)

        shapes = (batch_size, seq_len, dim_emb)
        use_time_emb = self.embed_use_time(batch["use_time_seq"]) if has_use_time else self.get_empty_emb(shapes)
        num_hint_emb = self.embed_num_hint(batch["num_hint_seq"]) if has_num_hint else self.get_empty_emb(shapes)
        num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"]) if has_num_attempt else self.get_empty_emb(shapes)
        interval_time_emb = self.embed_interval_time(batch["interval_time_seq"]) if has_time else self.get_empty_emb(shapes)
        aux_info_emb = self.aux_emb_fusion(torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1))
        interaction_emb = self.interaction_fusion(
            torch.cat((question_emb, concept_emb, correct_emb, aux_info_emb), dim=-1)
        )

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        latent_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        predict_score = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            question_emb_current = question_emb[:, t]
            concept_emb_current = concept_emb[:, t]
            correct_emb_current = correct_emb[:, t]
            interaction_emb_current = interaction_emb[:, t]
            interval_time_emb_current = interval_time_emb[:, t]

            # 从一道题中能学到多少只和题目有关
            learning_gain_current = torch.tanh(self.learning_gain_trans(
                torch.cat((question_emb_current, concept_emb_current, correct_emb_current), dim=-1)
            ))
            # 能吸收多少和学生做题情况（包括辅助信息）有关
            learning_absorb_gate = torch.sigmoid(self.gate4know_absorb(
                torch.cat((latent_pre, interaction_emb_current), dim=-1)
            ))
            learning_gain_current = learning_gain_current * learning_absorb_gate

            forget_gate = torch.sigmoid(self.gate4forget(
                torch.cat((latent_pre, interaction_emb_current, interval_time_emb_current), dim=-1)
            ))
            latent_current = latent_pre * forget_gate + learning_gain_current

            question_emb_next = question_emb[:, t + 1]
            question_next = question_seq[:, t + 1]
            predict_score_next = self.calculate_predict_score(latent_current, question_emb_next, question_next)
            predict_score[:, t+1] = predict_score_next
            latent[:, t+1, :] = latent_current
            latent_pre = latent_current

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]

    def get_predict_loss(self, batch, loss_record=None):
        loss = 0.
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        return loss
