import torch.nn.init

from .util import *
from .Module.KTEmbedLayer import KTEmbedLayer
from ..CONSTANT import HAS_TIME, HAS_USE_TIME, HAS_NUM_HINT, HAS_NUM_ATTEMPT


class MLP4Proj(nn.Module):
    def __init__(self, num_layer, dim_in, dim_out, dropout):
        super().__init__()

        self.num_layer = num_layer
        dim_middle = (dim_in + dim_out) // 2
        if num_layer == 1:
            self.mlp = nn.Linear(dim_in, dim_out)
        elif num_layer == 2:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, dim_out)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, dim_middle),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_middle, dim_out)
            )
        if num_layer > 2:
            self.residual = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.mlp(x) if (self.num_layer <= 2) else torch.relu(self.mlp(x) + self.residual(x))


class AuxInfoDCT(nn.Module):
    model_name = "AuxInfoDCT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dataset_name = encoder_config["dataset_name"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        num_mlp_layer = encoder_config["num_mlp_layer"]
        dropout = encoder_config["dropout"]

        self.has_time = dataset_name in HAS_TIME
        self.has_use_time = dataset_name in HAS_USE_TIME
        self.has_num_hint = dataset_name in HAS_NUM_HINT
        self.has_num_attempt = dataset_name in HAS_NUM_ATTEMPT

        # 输入embedding融合层（每种辅助信息的toke表不超过100， 在前端就处理好）
        self.embed_question = nn.Embedding(num_question+1, dim_emb)
        self.embed_concept = nn.Embedding(num_concept+1, dim_emb)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        torch.nn.init.xavier_uniform_(self.embed_concept.weight)
        if self.has_time:
            self.embed_interval_time = nn.Embedding(101, dim_emb)
        if self.has_use_time:
            self.embed_use_time = nn.Embedding(101, dim_emb)
        if self.has_num_hint:
            self.embed_num_hint = nn.Embedding(101, dim_emb)
        if self.has_num_attempt:
            self.embed_num_attempt = nn.Embedding(101, dim_emb)
        # 融合use time、num hint、num attempt
        self.fuse_ut_nh_na = nn.Linear(dim_emb * 3, dim_emb)

        # encode层：RNN
        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_emb * 5
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_emb * 4
        else:
            dim_rrn_input = dim_emb * 3
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)

        # question和latent的投影层
        self.que2difficulty = MLP4Proj(num_mlp_layer, dim_emb, num_concept, dropout)
        self.latent2ability = MLP4Proj(num_mlp_layer, dim_latent, num_concept, dropout)
        self.que2discrimination = MLP4Proj(num_mlp_layer, dim_emb, 1, dropout)
        self.dropout = nn.Dropout(dropout)

    def get_concept_emb(self, batch):
        use_mean_pool = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["use_mean_pool4concept"]
        if use_mean_pool:
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

    def predict_score(self, latent, question_emb, question_seq):
        max_que_disc = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["max_que_disc"]
        Q_table = self.objects["data"]["Q_table_tensor"]
        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))) * max_que_disc
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        user_ability_ = user_ability * Q_table[question_seq]
        que_difficulty_ = que_difficulty * Q_table[question_seq]
        # 使用补偿性模型，即对于多知识点习题，在考察的一个知识点上的不足可以由其它知识点补偿，同时考虑习题和知识点的关联强度
        sum_weight_concept = torch.sum(que_difficulty_, dim=-1, keepdim=True) + 1e-6
        irt_logits = que_discrimination * (user_ability_ - que_difficulty_)
        y = irt_logits / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def get_latent(self, batch, correct_noise_strength=0.):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_emb = encoder_config["dim_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)
        if 0 < correct_noise_strength < 0.5:
            noise = torch.rand(batch_size, seq_len, dim_emb).to(self.params["device"]) * correct_noise_strength
            noise_mask = torch.ones_like(noise).float().to(self.params["device"])
            noise_mask[correct_seq == 1] = -1
            noise = noise * noise_mask
            correct_emb = correct_emb + noise
        interaction_emb = torch.cat((question_emb, concept_emb, correct_emb), dim=2)

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            if self.has_use_time:
                use_time_emb = self.embed_use_time(batch["use_time_seq"])
            else:
                use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
            if self.has_num_hint:
                num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
            else:
                num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
            if self.has_num_attempt:
                num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
            else:
                num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])

            ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
            ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
            interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
            encoder_input = torch.cat((interaction_emb, interval_time_emb, ut_nh_na_emb), dim=-1)

        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            if self.has_time:
                interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
                encoder_input = torch.cat((interaction_emb, interval_time_emb), dim=-1)
            else:
                if self.has_use_time:
                    use_time_emb = self.embed_use_time(batch["use_time_seq"])
                else:
                    use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                if self.has_num_hint:
                    num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
                else:
                    num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                if self.has_num_attempt:
                    num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
                else:
                    num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
                ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
                encoder_input = torch.cat((interaction_emb, ut_nh_na_emb), dim=-1)

        else:
            encoder_input = interaction_emb

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(encoder_input)

        return latent

    def forward(self, batch):
        question_seq = batch["question_seq"]
        question_emb = self.embed_question(question_seq)
        latent = self.get_latent(batch)
        predict_score = self.predict_score(latent[:, :-1], question_emb[:, 1:], question_seq[:, 1:])

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_emb = encoder_config["dim_emb"]
        max_que_disc = encoder_config["max_que_disc"]

        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        data_type = self.params["datasets_config"]["data_type"]

        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]
        question_emb = self.embed_question(question_seq)
        concept_emb = self.get_concept_emb(batch)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        interaction_emb = torch.cat((question_emb[:, :-1], concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            if self.has_use_time:
                use_time_emb = self.embed_use_time(batch["use_time_seq"])
            else:
                use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
            if self.has_num_hint:
                num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
            else:
                num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
            if self.has_num_attempt:
                num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
            else:
                num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])

            ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
            ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
            interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
            encoder_input = torch.cat(
                (interaction_emb, interval_time_emb[:, :-1], ut_nh_na_emb[:, :-1]), dim=-1
            )

        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            if self.has_time:
                interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
                encoder_input = torch.cat((interaction_emb, interval_time_emb[:, :-1]), dim=-1)
            else:
                if self.has_use_time:
                    use_time_emb = self.embed_use_time(batch["use_time_seq"])
                else:
                    use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                if self.has_num_hint:
                    num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
                else:
                    num_hint_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                if self.has_num_attempt:
                    num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
                else:
                    num_attempt_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
                ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
                ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
                encoder_input = torch.cat((interaction_emb, ut_nh_na_emb[:, :-1]), dim=-1)

        else:
            encoder_input = interaction_emb

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(encoder_input)

        Q_table = self.objects["data"]["Q_table_tensor"]
        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb[:, 1:]))) * max_que_disc
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
        inter_func_in = user_ability - que_difficulty
        que_difficulty_ = que_difficulty * Q_table[question_seq[:, 1:]]
        irt_logits = que_discrimination * inter_func_in * Q_table[question_seq[:, 1:]]
        sum_weight_concept = torch.sum(que_difficulty_, dim=-1, keepdim=True) + 1e-6
        y = irt_logits / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        loss = 0.

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]
        if (not multi_stage) and (w_penalty_neg != 0):
            # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
            # 如果是单知识点数据集，那么对于做错的题，惩罚user_ability - que_difficulty大于0的值（只惩罚考察的知识点）
            target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table[:, 1:])
            mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                                 q2c_mask_table[:, 1:].bool()
            target_inter_func_in1 = torch.masked_select(target_inter_func_in, mask4inter_func_in)
            neg_inter_func_in = target_inter_func_in1[target_inter_func_in1 <= 0]
            num_sample = neg_inter_func_in.numel()

            if data_type == "single_concept":
                mask4inter_func_in2 = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                      (1 - batch["correct_seq"][:, 1:]).bool().unsqueeze(-1) & \
                                      q2c_mask_table[:, 1:].bool()
                target_inter_func_in2 = torch.masked_select(target_inter_func_in, mask4inter_func_in2)
                pos_inter_func_in = target_inter_func_in2[target_inter_func_in2 >= 0]
                num_sample = num_sample + pos_inter_func_in.numel()
                if num_sample > 0:
                    penalty_value = torch.cat((-neg_inter_func_in, pos_inter_func_in))
                    penalty_neg_loss = penalty_value.mean()
                    if loss_record is not None:
                        loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                             num_sample)
                    loss = loss + penalty_neg_loss * w_penalty_neg
            else:
                if num_sample > 0:
                    penalty_neg_loss = -neg_inter_func_in.mean()
                    if loss_record is not None:
                        loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                             num_sample)
                    loss = loss + penalty_neg_loss * w_penalty_neg

        return loss

    def get_penalty_neg_loss(self, batch):
        # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
        # 如果是单知识点数据集，那么对于做错的题，惩罚user_ability - que_difficulty大于0的值（只惩罚考察的知识点）
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]
        data_type = self.params["datasets_config"]["data_type"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        concept_emb = self.get_concept_emb(batch)
        interaction_emb = torch.cat((question_emb[:, :-1], concept_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
        inter_func_in = user_ability - que_difficulty

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]
        target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table[:, 1:])
        mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                             batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                             q2c_mask_table[:, 1:].bool()
        target_inter_func_in1 = torch.masked_select(target_inter_func_in, mask4inter_func_in)
        neg_inter_func_in = target_inter_func_in1[target_inter_func_in1 <= 0]
        num_sample = neg_inter_func_in.numel()

        if data_type == "single_concept":
            mask4inter_func_in2 = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 (1 - batch["correct_seq"][:, 1:]).bool().unsqueeze(-1) & \
                                 q2c_mask_table[:, 1:].bool()
            target_inter_func_in2 = torch.masked_select(target_inter_func_in, mask4inter_func_in2)
            pos_inter_func_in = target_inter_func_in2[target_inter_func_in2 >= 0]
            num_sample = num_sample + pos_inter_func_in.numel()
            if num_sample > 0:
                penalty_value = torch.cat((-neg_inter_func_in, pos_inter_func_in))
                return penalty_value.mean(), num_sample
            else:
                return 0, 0
        else:
            if num_sample > 0:
                return -neg_inter_func_in.mean(), num_sample
            else:
                return 0, 0
