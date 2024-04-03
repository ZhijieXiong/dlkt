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
        use_concept_input = encoder_config["use_concept_input"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        que_user_share_proj = encoder_config["que_user_share_proj"]
        num_mlp_layer = encoder_config["num_mlp_layer"]
        dropout = encoder_config["dropout"]

        self.has_time = dataset_name in HAS_TIME
        self.has_use_time = dataset_name in HAS_USE_TIME
        self.has_num_hint = dataset_name in HAS_NUM_HINT
        self.has_num_attempt = dataset_name in HAS_NUM_ATTEMPT

        # 输入embedding融合层（每种辅助信息的toke表不超过100， 在前端就处理好），+2是留出空位给virtual emb
        # 这里+5000是给virtual question（如果single concept对应的知识点数量为n，则该值为2n）留出空位，目前所有数据集该值都不超过5000
        self.embed_question = nn.Embedding(num_question + 5000, dim_question)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        if self.has_time:
            self.embed_interval_time = nn.Embedding(100 + 2, dim_question)
        if self.has_use_time:
            self.embed_use_time = nn.Embedding(100 + 2, dim_question)
        if self.has_num_hint:
            self.embed_num_hint = nn.Embedding(100 + 2, dim_question)
        if self.has_num_attempt:
            self.embed_num_attempt = nn.Embedding(100 + 2, dim_question)
        if use_concept_input:
            self.embed_concept = nn.Embedding(num_concept, dim_question)
            torch.nn.init.xavier_uniform_(self.embed_concept.weight)
            # 融合question、concept、correct
            self.fuse_q_c_c = nn.Linear(dim_question * 3, dim_question)
        # 融合use time、num hint、num attempt
        self.fuse_ut_nh_na = nn.Linear(dim_question * 3, dim_question)

        # encode层：RNN
        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * (4 - int(use_concept_input))
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * (3 - int(use_concept_input))
        else:
            dim_rrn_input = dim_question * (2 - int(use_concept_input))
        dim_rnn_output = dim_question if que_user_share_proj else dim_latent
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)

        # question和latent的投影层
        self.que2difficulty = MLP4Proj(num_mlp_layer, dim_question, num_concept, dropout)
        self.latent2ability = self.que2difficulty if que_user_share_proj else \
            MLP4Proj(num_mlp_layer, dim_latent, num_concept, dropout)
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
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
        question_emb = self.embed_question(batch_question)
        if test_theory == "rasch":
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
        else:
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))

        return que_difficulty

    def get_user_ability_init(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        use_concept_input = encoder_config["use_concept_input"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * (4 - int(use_concept_input))
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * (3 - int(use_concept_input))
        else:
            dim_rrn_input = dim_question * (2 - int(use_concept_input))

        h0 = torch.zeros(num_rnn_layer, 1, dim_latent, requires_grad=True).to(self.params["device"])
        latent, _ = self.encoder_layer(torch.zeros(1, 1, dim_rrn_input).to(self.params["device"]), h0)
        user_ability = torch.sigmoid(self.latent2ability(latent).squeeze(0).squeeze(0))

        return user_ability

    def predict_score(self, latent, question_emb, question_seq):
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
            y = (user_ability - que_difficulty) * self.objects["data"]["Q_table_tensor"][question_seq]
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
            que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))) * 10
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
            # mask掉太小的值
            que_diff_mask = torch.ones_like(que_difficulty).float().to(self.params["device"])
            que_diff_mask[que_difficulty < 0.05] = 0
            y = (que_discrimination * (user_ability - que_difficulty)) * \
                que_difficulty * que_diff_mask / torch.sum(que_difficulty, dim=1, keepdim=True)
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def get_latent(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_question = encoder_config["dim_question"]
        weight_aux_emb = encoder_config["weight_aux_emb"]
        use_concept_input = encoder_config["use_concept_input"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        if use_concept_input:
            concept_emb = self.get_concept_emb(batch)
            interaction_emb = self.fuse_q_c_c(torch.cat((question_emb, concept_emb, correct_emb), dim=2))
        else:
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
        use_concept_input = encoder_config["use_concept_input"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_cl_loss = self.params["loss_config"].get("cl loss", 0)
        data_type = self.params["datasets_config"]["data_type"]

        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]
        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        if use_concept_input:
            concept_emb = self.get_concept_emb(batch)
            q_c_c_emb = torch.cat((question_emb, concept_emb, correct_emb), dim=2)
            interaction_emb = self.fuse_q_c_c(q_c_c_emb[:, :-1])
        else:
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
        if test_theory == "rasch":
            concept_related = self.objects["data"]["Q_table_tensor"][question_seq[:, 1:]]
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb[:, 1:]))
            inter_func_in = user_ability - que_difficulty
            y = inter_func_in * concept_related
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
            que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb[:, 1:]))) * 10
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
            inter_func_in = user_ability - que_difficulty
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
            que_diff_mask = torch.ones_like(que_difficulty).float().to(self.params["device"])
            que_diff_mask[que_difficulty < 0.05] = 0
            y = (que_discrimination * inter_func_in) * \
                que_difficulty * que_diff_mask / torch.sum(que_difficulty, dim=1, keepdim=True)
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

        if (not multi_stage) and (w_cl_loss != 0):
            unbias_loss = self.get_unbias_loss(batch)
            if loss_record is not None:
                loss_record.add_loss("cl loss", unbias_loss.detach().cpu().item() * batch_size, batch_size)
            loss = loss + unbias_loss * w_cl_loss

        return loss

    def get_q_table_loss(self, target_question, question_ids, related_concept_ids, unrelated_concept_ids, t=0.5):
        # 根据数据集提供的Q table约束que2difficulty的学习
        # 一方面每道习题标注的知识点要比未标注的大；另一方面限制未标注的知识点小于一个阈值，如0.5
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        question_emb = self.embed_question(target_question)
        if test_theory == "rasch":
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
        else:
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        related_diff = que_difficulty[question_ids, related_concept_ids]
        unrelated_diff = que_difficulty[question_ids, unrelated_concept_ids]

        minus_diff = unrelated_diff - related_diff
        to_punish1 = minus_diff[minus_diff > 0]
        num_sample1 = to_punish1.numel()

        to_punish2 = unrelated_diff[unrelated_diff > t] - t
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
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
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

        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb[:, 1:]))
            inter_func_in = user_ability - que_difficulty
        else:
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

    def get_unbias_loss(self, batch):
        batch_original = {
            "question_seq": batch["question_seq"],
            "correct_seq": batch["correct_seq"],
            "mask_seq": batch["mask_seq"]
        }
        batch_aug = {
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        if "interval_time_seq" in batch.keys():
            batch_original["interval_time_seq"] = batch["interval_time_seq"]
            batch_aug["interval_time_seq"] = batch["interval_time_seq_aug_0"]
        if "use_time_seq" in batch.keys():
            batch_original["use_time_seq"] = batch["use_time_seq"]
            batch_aug["use_time_seq"] = batch["use_time_seq_aug_0"]
        if "use_time_first_seq" in batch.keys():
            batch_original["use_time_first_seq"] = batch["use_time_first_seq"]
            batch_aug["use_time_first_seq"] = batch["use_time_first_seq_aug_0"]
        if "num_hint_seq" in batch.keys():
            batch_original["num_hint_seq"] = batch["num_hint_seq"]
            batch_aug["num_hint_seq"] = batch["num_hint_seq_aug_0"]
        if "num_attempt_seq" in batch.keys():
            batch_original["num_attempt_seq"] = batch["num_attempt_seq"]
            batch_aug["num_attempt_seq"] = batch["num_attempt_seq_aug_0"]

        batch_size = batch["mask_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])

        latent_original = self.get_latent(batch_original)
        latent_original = latent_original[first_index, batch["seq_len_original"] - 1]

        latent_aug = self.get_latent(batch_aug)
        latent_aug = latent_aug[first_index, batch["seq_len_aug_0"] - 1]

        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim = torch.cosine_similarity(latent_original.unsqueeze(1), latent_aug.unsqueeze(0), dim=-1) / temp
        batch_size = cos_sim.size(0)
        labels = torch.arange(batch_size).long().to(self.params["device"])
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss
