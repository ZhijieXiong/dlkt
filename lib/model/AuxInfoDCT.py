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
        use_proj = encoder_config["use_proj"]
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
        dim_rnn_output = dim_latent if use_proj else num_concept
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)

        # question和latent的投影层
        self.que2difficulty = MLP4Proj(num_mlp_layer, dim_emb, num_concept, dropout)
        self.latent2ability = MLP4Proj(num_mlp_layer, dim_latent, num_concept, dropout)
        self.que2discrimination = MLP4Proj(num_mlp_layer, dim_emb, 1, dropout)
        self.dropout = nn.Dropout(dropout)

    def get_concept_emb(self, batch):
        use_mean_pool = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["use_mean_pool4concept"]
        data_type = self.params["datasets_config"]["data_type"]
        if use_mean_pool and data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        elif data_type == "single_concept":
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
        use_proj = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["use_proj"]
        Q_table = self.objects["data"]["Q_table_tensor"]

        if use_proj:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        else:
            user_ability = torch.sigmoid(latent)

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

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_emb = encoder_config["dim_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        concept_emb = self.get_concept_emb(batch)
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
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        use_proj = encoder_config["use_proj"]
        dim_emb = encoder_config["dim_emb"]
        max_que_disc = encoder_config["max_que_disc"]

        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_counter_fact = self.params["loss_config"].get("counterfactual loss", 0)
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_cl_loss = self.params["loss_config"].get("learning loss", 0)
        data_type = self.params["datasets_config"]["data_type"]

        Q_table = self.objects["data"]["Q_table_tensor"]

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

        # cf: counterfactual
        cf_user_ability = torch.zeros(batch_size, seq_len - 1, num_concept).to(self.params["device"])
        if (not multi_stage) and (w_counter_fact != 0):
            # 如果使用反事实约束，为了获取RNN每个时刻的hidden state，只能这么写
            # todo: 这里的代码可能有问题
            latent = torch.zeros(batch_size, seq_len - 1, dim_latent).to(self.params["device"])
            cf_correct_emb = (1 - correct_seq).reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, seq_len, -1)
            cf_interaction_emb = torch.cat((question_emb, cf_correct_emb), dim=2)
            # GRU 官方代码初始化h为0向量
            rnn_h_current = torch.zeros(
                num_rnn_layer, batch_size, dim_latent, dtype=encoder_input.dtype
            ).to(self.params["device"])
            for t in range(seq_len - 1):
                input_current = encoder_input[:, t].unsqueeze(1)
                latent_current, rnn_h_next = self.encoder_layer(input_current, rnn_h_current)
                latent[:, t] = latent_current.squeeze(1)

                cf_input_current = cf_interaction_emb[:, t].unsqueeze(1)
                cf_latent, _ = self.encoder_layer(cf_input_current, rnn_h_current)
                cf_user_ability[:, t] = torch.sigmoid(self.latent2ability(self.dropout(cf_latent.squeeze(1))))
                rnn_h_current = rnn_h_next
        else:
            # 如果不使用反事实约束
            self.encoder_layer.flatten_parameters()
            latent, _ = self.encoder_layer(encoder_input)

        if use_proj:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
        else:
            user_ability = torch.sigmoid(latent)

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
            # 单知识点习题：对于做对的题，惩罚user_ability - que_difficulty小于0的值
            #            对于做错的题，惩罚user_ability - que_difficulty大于0的值
            #            只惩罚考察的知识点
            if data_type == "single_concept":
                target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table[:, 1:])

                mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                     batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                                     q2c_mask_table[:, 1:].bool()
                target_inter_func_in1 = torch.masked_select(target_inter_func_in, mask4inter_func_in)
                neg_inter_func_in = target_inter_func_in1[target_inter_func_in1 <= 0]
                num_sample = neg_inter_func_in.numel()

                mask4inter_func_in2 = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                      (1 - batch["correct_seq"][:, 1:]).bool().unsqueeze(-1) & \
                                      q2c_mask_table[:, 1:].bool()
                target_inter_func_in2 = torch.masked_select(target_inter_func_in, mask4inter_func_in2)
                pos_inter_func_in = target_inter_func_in2[target_inter_func_in2 >= 0]
                num_sample = num_sample + pos_inter_func_in.numel()
            else:
                # 放弃的方案：多知识点习题：选择权重最重的惩罚，但是效果不好
                # qc_related_extent = torch.gather(que_difficulty, 2, q2c_table[:, 1:]) * q2c_mask_table[:, 1:]
                # qc_max_extent_index = torch.argmax(qc_related_extent, dim=2).unsqueeze(-1)
                # target_inter_func_in = torch.gather(inter_func_in, 2, qc_max_extent_index)

                # 方案1：对多知识点的习题损失乘一个小于1的权重，目前该方法最有效
                penalty_weight4question = self.objects["data"]["penalty_weight4question"][question_seq[:, 1:]]
                inter_func_in = inter_func_in * penalty_weight4question.unsqueeze(-1)

                # 方案2：不计算多知识点习题的损失
                # mask4single_concept = self.objects["data"]["mask4single_concept"][question_seq]
                # mask4single_concept[:, 1:].unsqueeze(-1) & \

                target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table[:, 1:])
                mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                     batch["correct_seq"][:, 1:].bool().unsqueeze(-1)
                target_inter_func_in1 = torch.masked_select(target_inter_func_in, mask4inter_func_in)
                neg_inter_func_in = target_inter_func_in1[target_inter_func_in1 <= 0]
                num_sample = neg_inter_func_in.numel()

                mask4inter_func_in2 = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                      (1 - batch["correct_seq"][:, 1:]).bool().unsqueeze(-1)
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

        if (not multi_stage) and (w_counter_fact != 0):
            # 反事实约束：做对一道题比做错一道题的学习增长大
            f_minus_cf = torch.gather(user_ability - cf_user_ability, 2, q2c_table[:, :-1])
            correct_seq1 = batch["correct_seq"][:, :-1].bool()
            correct_seq2 = (1 - batch["correct_seq"][:, :-1]).bool()
            mask4correct = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq1.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()
            mask4wrong = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq2.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()

            # 对于做对的时刻，f_minus_cf应该大于0，惩罚小于0的部分
            target_neg_f_minus_cf = torch.masked_select(f_minus_cf, mask4correct)
            neg_f_minus_cf = target_neg_f_minus_cf[target_neg_f_minus_cf < 0]
            num_sample1 = neg_f_minus_cf.numel()

            # 对于做错的时刻，f_minus_cf应该小于0，惩罚大于0的部分
            target_pos_f_minus_cf = torch.masked_select(f_minus_cf, mask4wrong)
            pos_f_minus_cf = target_pos_f_minus_cf[target_pos_f_minus_cf > 0]
            num_sample2 = pos_f_minus_cf.numel()

            num_sample = num_sample1 + num_sample2
            if num_sample > 0:
                cf_loss = torch.cat((-neg_f_minus_cf, pos_f_minus_cf)).mean()
                if loss_record is not None:
                    loss_record.add_loss("counterfactual loss", cf_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + cf_loss * w_counter_fact

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
            unbias_loss = self.get_unbiased_cl_loss(batch)
            if loss_record is not None:
                loss_record.add_loss("unbiased cl loss", unbias_loss.detach().cpu().item() * batch_size, batch_size)
            loss = loss + unbias_loss * w_cl_loss

        return loss

    def get_unbiased_cl_loss(self, batch):
        temp = self.params["other"]["instance_cl"]["temp"]
        correct_noise_strength = self.params["other"]["instance_cl"]["correct_noise_strength"]

        batch_size = batch["mask_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])

        latent_aug0 = self.get_latent(batch, correct_noise_strength)
        latent_aug0 = latent_aug0[first_index, batch["seq_len"] - 1]
        latent_aug1 = self.get_latent(batch, correct_noise_strength)
        latent_aug1 = latent_aug1[first_index, batch["seq_len"] - 1]

        cos_sim = torch.cosine_similarity(latent_aug0.unsqueeze(1), latent_aug1.unsqueeze(0), dim=-1) / temp
        batch_size = cos_sim.size(0)
        labels = torch.arange(batch_size).long().to(self.params["device"])
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_latent(self, batch, correct_noise_strength=0.):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]
        dim_emb = encoder_config["dim_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        question_emb = self.embed_question(question_seq)
        concept_emb = self.get_concept_emb(batch)
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        if 0 < correct_noise_strength < 0.5:
            # 均值噪声
            # noise = torch.rand(batch_size, seq_len, dim_emb).to(self.params["device"]) * correct_noise_strength
            # 高斯噪声
            noise = torch.normal(correct_noise_strength, 0.1, size=(batch_size, seq_len, dim_emb)).to(self.params["device"])

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
