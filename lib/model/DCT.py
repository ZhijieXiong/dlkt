import torch.nn.init

from .util import *
from .loss_util import binary_entropy


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
        dim_correct = encoder_config["dim_correct"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        que_user_share_proj = encoder_config["que_user_share_proj"]
        dropout = encoder_config["dropout"]

        self.embed_question = nn.Embedding(num_question, dim_question)
        dim_rnn_output = dim_question if que_user_share_proj else dim_latent
        dim_rrn_input = dim_question + dim_correct
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_rnn_output, batch_first=True, num_layers=num_rnn_layer)
        self.que2difficulty = nn.Linear(dim_question, num_concept)
        self.latent2ability = self.que2difficulty if que_user_share_proj else nn.Linear(dim_latent, num_concept)
        self.que2discrimination = nn.Linear(dim_question, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        que_user_share_proj = encoder_config["que_user_share_proj"]

        if not que_user_share_proj:
            torch.nn.init.xavier_uniform_(self.latent2ability.weight)

        torch.nn.init.xavier_uniform_(self.que2difficulty.weight)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)

        torch.nn.init.xavier_uniform_(self.que2discrimination.weight)

    def predict_score(self, latent, question_emb, question_seq):
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
            concept_related = self.objects["data"]["Q_table_tensor"][question_seq]
            y = (user_ability + que_difficulty) * concept_related
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
            que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb))) * 10
            y = (que_discrimination * (user_ability - que_difficulty)) * que_difficulty
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def forward(self, batch):
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
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
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        question_emb = self.embed_question(target_question)
        if test_theory == "rasch":
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
        else:
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
        related_diff = que_difficulty[question_ids, related_concept_ids]
        unrelated_diff = que_difficulty[question_ids, unrelated_concept_ids]

        minus_diff = unrelated_diff - related_diff
        to_punish = minus_diff[minus_diff > 0]
        num_sample = to_punish.numel()
        if num_sample > 0:
            return to_punish.mean(), num_sample
        else:
            return 0, 0

    def get_learn_loss(self, batch):
        # 学习约束：做了题比不做题学习增长大：对于每个时刻，都应该比上个时刻有所增长（对应当前时刻所做题的知识点），惩罚小于0的部分
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]

        master_leval = user_ability[:, 1:] - user_ability[:, :-1]
        mask4master = mask_bool_seq[:, 1:-1].unsqueeze(-1) & q2c_mask_table[:, 1:-1].bool()
        target_neg_master_leval = torch.masked_select(
            torch.gather(master_leval, 2, q2c_table[:, 1:-1]), mask4master
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
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb[:, 1:]))
            inter_func_in = user_ability + que_difficulty
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
        target_inter_func_in = torch.masked_select(target_inter_func_in, mask4inter_func_in)
        neg_inter_func_in = target_inter_func_in[target_inter_func_in <= 0]
        num_sample = neg_inter_func_in.numel()
        if num_sample > 0:
            penalty_neg_loss = -neg_inter_func_in.mean()
            return penalty_neg_loss, num_sample
        else:
            return 0, 0

    def get_counter_fact_loss(self, batch):
        # 反事实约束：做对一道题比做错一道题的学习增长大
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        cf_user_ability = torch.zeros(batch_size, seq_len - 1, num_concept).to(self.params["device"])
        latent = torch.zeros(batch_size, seq_len - 1, dim_latent).to(self.params["device"])
        cf_correct_emb = (1 - correct_seq).reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, seq_len, -1)
        cf_interaction_emb = torch.cat((question_emb, cf_correct_emb), dim=2)
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

        if test_theory == "rasch":
            user_ability = self.latent2ability(self.dropout(latent))
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))

        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"]]

        cf_loss = 0.
        f_minus_cf = torch.gather(user_ability - cf_user_ability, 2, q2c_table[:, :-1])
        correct_seq1 = batch["correct_seq"][:, :-1].bool()
        correct_seq2 = (1 - batch["correct_seq"][:, :-1]).bool()
        mask4correct = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq1.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()
        mask4wrong = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq2.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()

        # 对于做对的时刻，f_minus_cf应该大于0，惩罚小于0的部分
        target_neg_f_minus_cf = torch.masked_select(f_minus_cf, mask4correct)
        neg_f_minus_cf = target_neg_f_minus_cf[target_neg_f_minus_cf < 0]
        num_sample1 = neg_f_minus_cf.numel()
        if num_sample1 > 0:
            cf_loss1 = -neg_f_minus_cf.mean()
            cf_loss = cf_loss + cf_loss1

        # 对于做错的时刻，f_minus_cf应该小于0，惩罚大于0的部分
        target_pos_f_minus_cf = torch.masked_select(f_minus_cf, mask4wrong)
        pos_f_minus_cf = target_pos_f_minus_cf[target_pos_f_minus_cf > 0]
        num_sample2 = pos_f_minus_cf.numel()
        if num_sample2 > 0:
            cf_loss2 = pos_f_minus_cf.mean()
            cf_loss = cf_loss + cf_loss2

        if (num_sample1 + num_sample2) > 0:
            return cf_loss, (num_sample1 + num_sample2)
        else:
            return 0, 0

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]
        dim_correct = encoder_config["dim_correct"]
        num_concept = encoder_config["num_concept"]
        dim_latent = encoder_config["dim_latent"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        w_penalty_neg = self.params["loss_config"].get("penalty neg loss", 0)
        w_learning = self.params["loss_config"].get("learning loss", 0)
        w_counter_fact = self.params["loss_config"].get("counterfactual loss", 0)
        multi_stage = self.params["other"]["cognition_tracing"]["multi_stage"]
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = self.embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        # cf: counterfactual
        cf_user_ability = torch.zeros(batch_size, seq_len - 1, num_concept).to(self.params["device"])
        if (not multi_stage) and (w_counter_fact != 0):
            # 如果使用反事实约束，为了获取RNN每个时刻的hidden state，只能这么写
            latent = torch.zeros(batch_size, seq_len - 1, dim_latent).to(self.params["device"])
            cf_correct_emb = (1 - correct_seq).reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, seq_len, -1)
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

        if test_theory == "rasch":
            concept_related = self.objects["data"]["Q_table_tensor"][question_seq[:, 1:]]
            user_ability = self.latent2ability(self.dropout(latent))
            que_difficulty = self.que2difficulty(self.dropout(question_emb[:, 1:]))
            inter_func_in = user_ability + que_difficulty
            y = inter_func_in * concept_related
        else:
            user_ability = torch.sigmoid(self.latent2ability(self.dropout(latent)))
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb[:, 1:])))
            inter_func_in = user_ability - que_difficulty
            que_discrimination = torch.sigmoid(self.que2discrimination(self.dropout(question_emb[:, 1:]))) * 10
            y = (que_discrimination * inter_func_in) * que_difficulty

        predict_score = torch.sigmoid(torch.sum(y, dim=-1))
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        loss = 0.
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
            target_inter_func_in = torch.gather(inter_func_in, 2, q2c_table[:, 1:])
            mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                                 q2c_mask_table[:, 1:].bool()
            target_inter_func_in = torch.masked_select(target_inter_func_in, mask4inter_func_in)
            neg_inter_func_in = target_inter_func_in[target_inter_func_in <= 0]
            if neg_inter_func_in.numel() > 0:
                penalty_neg_loss = -neg_inter_func_in.mean()
                if loss_record is not None:
                    num_sample = neg_inter_func_in.shape[0]
                    loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample,
                                         num_sample)
                loss = loss + penalty_neg_loss * w_penalty_neg

        if (not multi_stage) and (w_learning != 0):
            # 学习约束：做了题比不做题学习增长大
            master_leval = user_ability[:, 1:] - user_ability[:, :-1]
            mask4master = mask_bool_seq[:, 1:-1].unsqueeze(-1) & q2c_mask_table[:, 1:-1].bool()
            # 对于每个时刻，都应该比上个时刻有所增长（对应当前时刻所做题的知识点），惩罚小于0的部分
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
            cf_loss = 0.

            f_minus_cf = torch.gather(user_ability - cf_user_ability, 2, q2c_table[:, :-1])
            correct_seq1 = batch["correct_seq"][:, :-1].bool()
            correct_seq2 = (1 - batch["correct_seq"][:, :-1]).bool()
            mask4correct = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq1.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()
            mask4wrong = mask_bool_seq[:, :-1].unsqueeze(-1) & correct_seq2.unsqueeze(-1) & q2c_mask_table[:, :-1].bool()

            # 对于做对的时刻，f_minus_cf应该大于0，惩罚小于0的部分
            target_neg_f_minus_cf = torch.masked_select(f_minus_cf, mask4correct)
            neg_f_minus_cf = target_neg_f_minus_cf[target_neg_f_minus_cf < 0]
            num_sample1 = neg_f_minus_cf.numel()
            if num_sample1 > 0:
                cf_loss1 = -neg_f_minus_cf.mean()
                cf_loss = cf_loss + cf_loss1
                if loss_record is not None:
                    loss_record.add_loss("counterfactual loss", cf_loss1.detach().cpu().item() * num_sample1, num_sample1)

            # 对于做错的时刻，f_minus_cf应该小于0，惩罚大于0的部分
            target_pos_f_minus_cf = torch.masked_select(f_minus_cf, mask4wrong)
            pos_f_minus_cf = target_pos_f_minus_cf[target_pos_f_minus_cf > 0]
            num_sample2 = pos_f_minus_cf.numel()
            if num_sample2 > 0:
                cf_loss2 = pos_f_minus_cf.mean()
                cf_loss = cf_loss + cf_loss2
                if loss_record is not None:
                    loss_record.add_loss("counterfactual loss", cf_loss2.detach().cpu().item() * num_sample2, num_sample2)

            if (num_sample1 + num_sample2) > 0:
                loss = loss + cf_loss * w_counter_fact

        return loss

    # --------------------------------------------------ME-ADA----------------------------------------------------------

    def forward_from_adv_data(self, dataset, batch):
        embed_question = dataset["embed_question"]
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = embed_question(question_seq)
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        predict_score = self.predict_score(latent, question_emb[:, 1:], question_seq[:, 1:])

        return predict_score

    def get_latent_from_adv_data(self, dataset, batch):
        embed_question = dataset["embed_question"]
        dim_correct = self.params["models_config"]["kt_model"]["encoder_layer"]["DCT"]["dim_correct"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        question_emb = embed_question(question_seq)
        interaction_emb = torch.cat((question_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        return latent

    def get_latent_last_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_predict_score_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward_from_adv_data(dataset, batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score_from_adv_data(dataset, batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        loss = predict_loss

        return loss

    def max_entropy_adv_aug(self, dataset, batch, optimizer, loop_adv, eta, gamma):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        latent_ori = self.get_latent_from_adv_data(dataset, batch).detach().clone()
        latent_ori = latent_ori[mask_bool_seq]
        latent_ori.requires_grad_(False)
        adv_predict_loss = 0.
        adv_entropy = 0.
        adv_mse_loss = 0.
        for ite_max in range(loop_adv):
            predict_score = self.get_predict_score_from_adv_data(dataset, batch)
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
            entropy_loss = binary_entropy(predict_score)
            latent = self.get_latent_from_adv_data(dataset, batch)
            latent = latent[mask_bool_seq]
            latent_mse_loss = nn.functional.mse_loss(latent, latent_ori)

            if ite_max == (loop_adv - 1):
                adv_predict_loss += predict_loss.detach().cpu().item()
                adv_entropy += entropy_loss.detach().cpu().item()
                adv_mse_loss += latent_mse_loss.detach().cpu().item()
            loss = predict_loss + eta * entropy_loss - gamma * latent_mse_loss
            self.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

        return adv_predict_loss, adv_entropy, adv_mse_loss
