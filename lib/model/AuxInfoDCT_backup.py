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
        dim_question = encoder_config["dim_question"]
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
        self.embed_question = nn.Embedding(num_question+1, dim_question)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        if self.has_time:
            self.embed_interval_time = nn.Embedding(101, dim_question)
        if self.has_use_time:
            self.embed_use_time = nn.Embedding(101, dim_question)
        if self.has_num_hint:
            self.embed_num_hint = nn.Embedding(101, dim_question)
        if self.has_num_attempt:
            self.embed_num_attempt = nn.Embedding(101, dim_question)
        # 融合use time、num hint、num attempt
        self.fuse_ut_nh_na = nn.Linear(dim_question * 3, dim_question)

        # encode层：RNN
        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * 4
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * 3
        else:
            dim_rrn_input = dim_question * 2
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)

        # question和latent的投影层
        self.que2difficulty = MLP4Proj(num_mlp_layer, dim_question, num_concept, dropout)
        self.latent2ability = MLP4Proj(num_mlp_layer, dim_latent, num_concept, dropout)
        self.que2discrimination = MLP4Proj(num_mlp_layer, dim_question, 1, dropout)
        self.dropout = nn.Dropout(dropout)

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

    def get_question_diff(self, batch_question):
        question_emb = self.embed_question(batch_question)
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))

        return que_difficulty

    def get_user_ability_init(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * 4
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * 3
        else:
            dim_rrn_input = dim_question * 2

        h0 = torch.zeros(num_rnn_layer, 1, dim_latent, requires_grad=True).to(self.params["device"])
        latent, _ = self.encoder_layer(torch.zeros(1, 1, dim_rrn_input).to(self.params["device"]), h0)
        user_ability = torch.sigmoid(self.latent2ability(latent).squeeze(0).squeeze(0))

        return user_ability

    def predict_score(self, latent, question_emb, question_seq):
        use_hard_Q_table = self.params["other"]["cognition_tracing"]["use_hard_Q_table"]
        Q_table = self.objects["data"]["Q_table_tensor"]
        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))) * 10
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))

        if use_hard_Q_table:
            # 使用原始Q table
            user_ability_ = user_ability * Q_table[question_seq]
            que_difficulty_ = que_difficulty * Q_table[question_seq]
        else:
            # 使用学习的Q table，mask掉太小的值
            que_diff_mask = torch.ones_like(que_difficulty).float().to(self.params["device"])
            que_diff_mask[que_difficulty < 0.05] = 0
            user_ability_ = user_ability * que_diff_mask
            que_difficulty_ = que_difficulty * que_diff_mask

        # 使用补偿性模型，即对于多知识点习题，在考察的一个知识点上的不足可以由其它知识点补偿，同时考虑习题和知识点的关联强度
        sum_weight_concept = torch.sum(que_difficulty_, dim=-1, keepdim=True) + 1e-6
        irt_logits = que_discrimination * (user_ability_ - que_difficulty_)
        y = irt_logits / sum_weight_concept
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def get_latent(self, batch, correct_noise_strength=0.):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_question = encoder_config["dim_question"]
        weight_aux_emb = encoder_config["weight_aux_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        if 0 < correct_noise_strength < 0.5:
            noise = torch.rand(batch_size, seq_len, dim_question).to(self.params["device"]) * correct_noise_strength
            noise_mask = torch.ones_like(noise).float().to(self.params["device"])
            noise_mask[correct_seq == 1] = -1
            noise = noise * noise_mask
            correct_emb = correct_emb + noise
        interaction_emb = torch.cat((question_emb, correct_emb), dim=2)

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            if self.has_use_time:
                use_time_emb = self.embed_use_time(batch["use_time_seq"])
            else:
                use_time_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
            if self.has_num_hint:
                num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
            else:
                num_hint_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
            if self.has_num_attempt:
                num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
            else:
                num_attempt_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])

            ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
            ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
            interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
            encoder_input = torch.cat((interaction_emb, interval_time_emb, ut_nh_na_emb * weight_aux_emb), dim=-1)

        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            if self.has_time:
                interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
                encoder_input = torch.cat((interaction_emb, interval_time_emb), dim=-1)
            else:
                if self.has_use_time:
                    use_time_emb = self.embed_use_time(batch["use_time_seq"])
                else:
                    use_time_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                if self.has_num_hint:
                    num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
                else:
                    num_hint_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                if self.has_num_attempt:
                    num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
                else:
                    num_attempt_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
                ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
                encoder_input = torch.cat((interaction_emb, ut_nh_na_emb * weight_aux_emb), dim=-1)

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
        dim_question = encoder_config["dim_question"]
        weight_aux_emb = encoder_config["weight_aux_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        use_hard_Q_table = self.params["other"]["cognition_tracing"]["use_hard_Q_table"]
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_cl_loss = self.params["loss_config"].get("cl loss", 0)
        data_type = self.params["datasets_config"]["data_type"]

        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]
        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            if self.has_use_time:
                use_time_emb = self.embed_use_time(batch["use_time_seq"])
            else:
                use_time_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
            if self.has_num_hint:
                num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
            else:
                num_hint_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
            if self.has_num_attempt:
                num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
            else:
                num_attempt_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])

            ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
            ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
            interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
            encoder_input = torch.cat(
                (interaction_emb, interval_time_emb[:, :-1], ut_nh_na_emb[:, :-1] * weight_aux_emb), dim=-1
            )

        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            if self.has_time:
                interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
                encoder_input = torch.cat((interaction_emb, interval_time_emb[:, :-1]), dim=-1)
            else:
                if self.has_use_time:
                    use_time_emb = self.embed_use_time(batch["use_time_seq"])
                else:
                    use_time_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                if self.has_num_hint:
                    num_hint_emb = self.embed_num_hint(batch["num_hint_seq"])
                else:
                    num_hint_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                if self.has_num_attempt:
                    num_attempt_emb = self.embed_num_hint(batch["num_attempt_seq"])
                else:
                    num_attempt_emb = torch.zeros(batch_size, seq_len, dim_question).to(self.params["device"])
                ut_nh_na_emb = torch.cat((use_time_emb, num_hint_emb, num_attempt_emb), dim=-1)
                ut_nh_na_emb = self.fuse_ut_nh_na(ut_nh_na_emb)
                encoder_input = torch.cat((interaction_emb, ut_nh_na_emb[:, :-1] * weight_aux_emb), dim=-1)

        else:
            encoder_input = interaction_emb

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(encoder_input)

        Q_table = self.objects["data"]["Q_table_tensor"]
        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb[:, 1:]))) * 10
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
        inter_func_in = user_ability - que_difficulty
        if use_hard_Q_table:
            que_difficulty_ = que_difficulty * Q_table[question_seq[:, 1:]]
            irt_logits = que_discrimination * inter_func_in * Q_table[question_seq[:, 1:]]
        else:
            que_diff_mask = torch.ones_like(que_difficulty).float().to(self.params["device"])
            que_diff_mask[que_difficulty < 0.05] = 0
            que_difficulty_ = que_difficulty * que_diff_mask
            irt_logits = que_discrimination * inter_func_in * que_diff_mask
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

        if (not multi_stage) and (w_learning != 0):
            # 学习约束（单调理论）：做对了题比不做题学习增长大
            master_leval = user_ability[:, 1:] - user_ability[:, :-1]
            mask4master = mask_bool_seq[:, 1:-1].unsqueeze(-1) & \
                          correct_seq[:, 1:-1].unsqueeze(-1).bool() & \
                          q2c_mask_table[:, 1:-1].bool()
            target_neg_master_leval = torch.masked_select(
                torch.gather(master_leval, 2, q2c_table[:, 1:-1]), mask4master
            )
            neg_master_leval = target_neg_master_leval[target_neg_master_leval < 0]
            num_sample = neg_master_leval.numel()
            if num_sample > 0:
                learn_loss = -neg_master_leval.mean()
                if loss_record is not None:
                    loss_record.add_loss("learning loss", learn_loss.detach().cpu().item() * num_sample,
                                         num_sample)
                loss = loss + learn_loss * w_learning

        if (not multi_stage) and (w_cl_loss != 0):
            unbias_loss = self.get_unbias_loss(batch)
            if loss_record is not None:
                loss_record.add_loss("cl loss", unbias_loss.detach().cpu().item() * batch_size, batch_size)
            loss = loss + unbias_loss * w_cl_loss

        return loss

    def get_q_table_loss(self, target_question, question_ids, related_concept_ids, unrelated_concept_ids):
        # 根据数据集提供的Q table约束que2difficulty的学习
        # 一方面每道习题标注的知识点要比未标注的大；另一方面限制未标注的知识点小于一个阈值，如0.5
        threshold = self.params["other"]["cognition_tracing"]["q_table_loss_th"]
        question_emb = self.embed_question(target_question)
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        related_diff = que_difficulty[question_ids, related_concept_ids]
        unrelated_diff = que_difficulty[question_ids, unrelated_concept_ids]

        minus_diff = unrelated_diff - related_diff
        to_punish1 = minus_diff[minus_diff > 0]
        num_sample1 = to_punish1.numel()

        to_punish2 = unrelated_diff[unrelated_diff > threshold] - threshold
        num_sample2 = to_punish2.numel()

        num_sample = num_sample1 + num_sample2
        if num_sample > 0:
            to_punish = torch.cat((to_punish1, to_punish2))
            return to_punish.mean(), num_sample
        else:
            return 0, 0

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

    def get_learn_loss(self, batch):
        # 学习约束：做对了题比不做题学习增长大
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]

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

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]

        master_leval = user_ability[:, 1:] - user_ability[:, :-1]
        mask4master = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                      correct_seq[:, 1:].unsqueeze(-1).bool() & \
                      q2c_mask_table[:, 1:].bool()
        target_neg_master_leval = torch.masked_select(
            torch.gather(master_leval, 2, q2c_table[:, 1:]), mask4master
        )
        neg_master_leval = target_neg_master_leval[target_neg_master_leval < 0]
        num_sample = neg_master_leval.numel()
        if num_sample > 0:
            learn_loss = -neg_master_leval.mean()
            return learn_loss, num_sample
        else:
            return 0, 0

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
