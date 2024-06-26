import torch.nn.init

from .util import *


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


class DCT(nn.Module):
    model_name = "DCT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        num_mlp_layer = encoder_config["num_mlp_layer"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question, dim_question)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        dim_rrn_input = dim_question * 2
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.que2difficulty = MLP4Proj(num_mlp_layer, dim_question, num_concept, dropout)
        self.latent2ability = MLP4Proj(num_mlp_layer, dim_latent, num_concept, dropout)
        self.que2discrimination = MLP4Proj(num_mlp_layer, dim_question, 1, dropout)
        self.dropout = nn.Dropout(dropout)

    def get_question_diff(self, batch_question):
        question_emb = self.embed_question(batch_question)
        que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))

        return que_difficulty

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

    def forward(self, batch):
        dim_question = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_question"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        predict_score = self.predict_score(latent, question_emb[:, 1:], question_seq[:, 1:])

        return predict_score

    def get_que_diff_pred_loss(self, target_question):
        Q_table_mask = self.objects["cognition_tracing"]["Q_table_mask"]
        que_diff_label = self.objects["cognition_tracing"]["que_diff_ground_truth"]
        mask = Q_table_mask[target_question].bool()

        question_emb = self.embed_question(target_question)
        pred_diff_all = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        pred_diff = torch.masked_select(pred_diff_all, mask)
        ground_truth = torch.masked_select(que_diff_label[target_question], mask)
        predict_loss = nn.functional.mse_loss(pred_diff, ground_truth)

        return predict_loss

    def get_que_disc_pred_loss(self, target_question):
        question_emb = self.embed_question(target_question)
        pred_disc = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))).squeeze(dim=-1) * 10
        ground_truth = self.objects["cognition_tracing"]["que_disc_ground_truth"]
        predict_loss = nn.functional.mse_loss(pred_disc, ground_truth)

        return predict_loss

    def get_q_table_loss(self, target_question, question_ids, related_concept_ids, unrelated_concept_ids):
        # 根据数据集提供的Q table约束que2difficulty的学习
        # 一方面每道习题标注的知识点要比未标注的大；另一方面限制未标注的知识点小于一个阈值，如0.2
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

    def get_learn_loss(self, batch):
        # 学习约束：做对了题比不做题学习增长大
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_question = encoder_config["dim_question"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb, correct_emb), dim=2)

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

    def get_penalty_neg_loss(self, batch):
        # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
        # 如果是单知识点数据集，那么对于做错的题，惩罚user_ability - que_difficulty大于0的值（只惩罚考察的知识点）
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_question = encoder_config["dim_question"]
        data_type = self.params["datasets_config"]["data_type"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

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

    def get_counter_fact_loss(self, batch):
        # 反事实约束：做对一道题比做错一道题的学习增长大
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_question = encoder_config["dim_question"]
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb, correct_emb), dim=2)

        cf_user_ability = torch.zeros(batch_size, seq_len, num_concept).to(self.params["device"])
        latent = torch.zeros(batch_size, seq_len, dim_latent).to(self.params["device"])
        cf_correct_emb = (1 - correct_seq).reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, seq_len, -1)
        cf_interaction_emb = torch.cat((question_emb, cf_correct_emb), dim=2)

        # torch中rnn的hidden state初始值为0
        rnn_h_current = torch.zeros(
            num_rnn_layer, batch_size, dim_latent, dtype=interaction_emb.dtype
        ).to(self.params["device"])
        for t in range(seq_len):
            input_current = interaction_emb[:, t].unsqueeze(1)
            latent_current, rnn_h_next = self.encoder_layer(input_current, rnn_h_current)
            latent[:, t] = latent_current.squeeze(1)

            cf_input_current = cf_interaction_emb[:, t].unsqueeze(1)
            cf_latent, _ = self.encoder_layer(cf_input_current, rnn_h_current)
            cf_user_ability[:, t] = torch.sigmoid(self.latent2ability(self.dropout(cf_latent.squeeze(1))))

            rnn_h_current = rnn_h_next

        user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]

        f_minus_cf = torch.gather(user_ability - cf_user_ability, 2, q2c_table)
        mask4correct = mask_bool_seq.unsqueeze(-1) & \
                       batch["correct_seq"].bool().unsqueeze(-1) & \
                       q2c_mask_table.bool()
        mask4wrong = mask_bool_seq.unsqueeze(-1) & \
                     (1 - batch["correct_seq"]).bool().unsqueeze(-1) & \
                     q2c_mask_table.bool()

        # 对于做对的时刻，f_minus_cf应该大于0，惩罚小于0的部分
        target_neg_f_minus_cf = torch.masked_select(f_minus_cf, mask4correct)
        neg_f_minus_cf = target_neg_f_minus_cf[target_neg_f_minus_cf < 0]
        num_sample1 = neg_f_minus_cf.numel()

        # 对于做错的时刻，f_minus_cf应该小于0，惩罚大于0的部分
        target_pos_f_minus_cf = torch.masked_select(f_minus_cf, mask4wrong)
        pos_f_minus_cf = target_pos_f_minus_cf[target_pos_f_minus_cf > 0]
        num_sample2 = pos_f_minus_cf.numel()

        if (num_sample1 + num_sample2) > 0:
            cf_value = torch.cat((-neg_f_minus_cf, pos_f_minus_cf))
            return cf_value.mean(), (num_sample1 + num_sample2)
        else:
            return 0, 0

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_question = encoder_config["dim_question"]
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_counter_fact = self.params["loss_config"].get("counterfactual loss", 0)
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        use_hard_Q_table = self.params["other"]["cognition_tracing"]["use_hard_Q_table"]
        data_type = self.params["datasets_config"]["data_type"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        # cf: counterfactual
        cf_user_ability = torch.zeros(batch_size, seq_len - 1, num_concept).to(self.params["device"])
        if (not multi_stage) and (w_counter_fact != 0):
            # 如果使用反事实约束，为了获取RNN每个时刻的hidden state，只能这么写
            latent = torch.zeros(batch_size, seq_len - 1, dim_latent).to(self.params["device"])
            cf_correct_emb = (1 - correct_seq).reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, seq_len, -1)
            cf_interaction_emb = torch.cat((question_emb, cf_correct_emb), dim=2)
            # GRU 官方代码初始化h为0向量
            rnn_h_current = torch.zeros(
                num_rnn_layer, batch_size, dim_latent, dtype=interaction_emb.dtype
            ).to(self.params["device"])
            for t in range(seq_len - 1):
                input_current = interaction_emb[:, t].unsqueeze(1)
                latent_current, rnn_h_next = self.encoder_layer(input_current, rnn_h_current)
                latent[:, t] = latent_current.squeeze(1)

                cf_input_current = cf_interaction_emb[:, t].unsqueeze(1)
                cf_latent, _ = self.encoder_layer(cf_input_current, rnn_h_current)
                cf_user_ability[:, t] = torch.sigmoid(self.latent2ability(self.dropout(cf_latent.squeeze(1))))

                rnn_h_current = rnn_h_next
        else:
            # 如果不使用反事实约束
            self.encoder_layer.flatten_parameters()
            latent, _ = self.encoder_layer(interaction_emb)

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
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]
        if (not multi_stage) and (w_penalty_neg != 0):
            # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
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
                        loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample, num_sample)
                    loss = loss + penalty_neg_loss * w_penalty_neg
            else:
                if num_sample > 0:
                    penalty_neg_loss = -neg_inter_func_in.mean()
                    if loss_record is not None:
                        loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                             num_sample)
                    loss = loss + penalty_neg_loss * w_penalty_neg

        if (not multi_stage) and (w_learning != 0):
            # 学习约束：做对了题比不做题学习增长大
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

        return loss
