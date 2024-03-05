import torch.nn.init

from .util import *


class LPKTPlusV1(nn.Module):
    model_name = "LPKTPlus"

    def __init__(self, params, objects):
        super(LPKTPlusV1, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        num_question = encoder_config["num_question"]
        num_interval_time = encoder_config["num_interval_time"]
        num_use_time = encoder_config["num_use_time"]
        dim_correct = encoder_config["dim_correct"]
        dim_e = encoder_config["dim_e"]
        dim_k = encoder_config["dim_k"]
        dropout = encoder_config["dropout"]
        ablation_set = encoder_config["ablation_set"]

        if ablation_set == 0:
            self.embed_answer_time = nn.Embedding(num_use_time + 1, dim_k)
            torch.nn.init.xavier_uniform_(self.embed_answer_time.weight)
        if ablation_set == 0 or ablation_set == 1:
            self.embed_interval_time = nn.Embedding(num_interval_time + 1, dim_k)
            torch.nn.init.xavier_uniform_(self.embed_interval_time.weight)
        self.embed_question = nn.Embedding(num_question + 1, dim_k)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)

        if ablation_set == 0:
            self.linear_1 = nn.Linear(dim_correct + dim_e + dim_k, dim_k)
        elif ablation_set == 1:
            self.linear_1 = nn.Linear(dim_correct + dim_e, dim_k)
        else:
            raise NotImplementedError()

        self.linear_2 = nn.Linear(4 * dim_k, dim_k)
        self.linear_3 = nn.Linear(4 * dim_k, dim_k)
        self.linear_4 = nn.Linear(3 * dim_k, dim_k)
        self.linear_5 = nn.Linear(dim_e + dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------base--------------------------------------------------------

    def forward(self, batch):
        question_seq = batch["question_seq"]
        interval_time_seq = batch["interval_time_seq"]
        correct_seq = batch["correct_seq"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        num_concept = encoder_config["num_concept"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        ablation_set = encoder_config["ablation_set"]
        q_matrix = self.objects["lpkt_plus"]["q_matrix"]

        batch_size, seq_len = question_seq.size(0), question_seq.size(1)
        question_emb = self.embed_question(question_seq)

        interval_time_emb = self.embed_interval_time(interval_time_seq)
        correct_emb = correct_seq.view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        if ablation_set == 0:
            use_time_seq = batch["use_time_seq"]
            use_time_emb = self.embed_answer_time(use_time_seq)
            all_learning = self.linear_1(torch.cat((question_emb, use_time_emb, correct_emb), 2))
        else:
            all_learning = self.linear_1(torch.cat((question_emb, correct_emb), 2))
        h_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_k)).repeat(batch_size, 1, 1).to(self.params["device"])
        h_tilde_pre = None
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])
        pred = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            e = question_seq[:, t]
            # q_e: (bs, 1, n_skill)
            q_e = q_matrix[e].view(batch_size, 1, -1)
            it = interval_time_emb[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, dim_k)
            learning = all_learning[:, t]
            learning_gain = self.tanh(self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1)))
            gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1)))
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, dim_k),
                it.repeat(1, n_skill).view(batch_size, -1, dim_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = q_matrix[question_seq[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, dim_k)
            y = self.sig(self.linear_5(torch.cat((question_emb[:, t + 1], h_tilde), 1))).sum(1) / dim_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss


class LPKTPlusV2(nn.Module):
    model_name = "LPKTPlus"

    def __init__(self, params, objects):
        super(LPKTPlusV2, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        num_concept = encoder_config["num_concept"]
        num_question = encoder_config["num_question"]
        num_interval_time = encoder_config["num_interval_time"]
        num_use_time = encoder_config["num_use_time"]
        dim_correct = encoder_config["dim_correct"]
        dim_e = encoder_config["dim_e"]
        dim_k = encoder_config["dim_k"]
        dropout = encoder_config["dropout"]
        ablation_set = encoder_config["ablation_set"]

        self.embed_question = nn.Embedding(num_question + 1, dim_k)
        if ablation_set == 0:
            self.embed_answer_time = nn.Embedding(num_use_time + 1, dim_k)
            self.embed_interval_time = nn.Embedding(num_interval_time + 1, dim_k)
            self.linear_1 = nn.Linear(dim_correct + dim_e + dim_k, dim_k)
        elif ablation_set == 1:
            self.embed_interval_time = nn.Embedding(num_interval_time + 1, dim_k)
            self.linear_1 = nn.Linear(dim_correct + dim_e, dim_k)
        else:
            raise NotImplementedError()
        self.linear_2 = nn.Linear(4 * dim_k, dim_k)
        self.linear_3 = nn.Linear(4 * dim_k, dim_k)
        self.linear_4 = nn.Linear(3 * dim_k, dim_k)
        self.proj_latent2ability = nn.Linear(dim_k, num_concept)
        self.proj_que2difficulty = nn.Linear(dim_k, num_concept)
        self.proj_que2discrimination = nn.Linear(dim_k, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        ablation_set = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]["ablation_set"]
        use_init_weight = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]["use_init_weight"]

        if ablation_set == 0:
            torch.nn.init.xavier_uniform_(self.embed_answer_time.weight)
        if ablation_set == 0 or ablation_set == 1:
            torch.nn.init.xavier_uniform_(self.embed_interval_time.weight)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)

        if use_init_weight:
            self.proj_latent2ability.weight = nn.Parameter(self.objects["lpkt_plus"]["user_proj_weight_init_value"])
            torch.nn.init.constant_(self.proj_latent2ability.bias, 0)
        else:
            torch.nn.init.xavier_uniform_(self.proj_latent2ability.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2difficulty.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2discrimination.weight)

    # ------------------------------------------------------base--------------------------------------------------------
    def predict_score(self, latent, question_emb, target_question):
        batch_size = latent.shape[0]
        user_ability = torch.sigmoid(self.proj_latent2ability(latent))
        que_difficulty = torch.sigmoid(self.proj_que2difficulty(question_emb))
        que_discrimination = torch.sigmoid(self.proj_que2discrimination(question_emb)) * 10
        target_que_concept = self.objects["lpkt_plus"]["q_matrix"][target_question].view(batch_size, -1, 1)
        y = (que_discrimination * (user_ability - que_difficulty)).view(batch_size, 1, -1)
        predict_score = torch.sigmoid(y.bmm(target_que_concept).squeeze(dim=-1).squeeze(dim=-1))

        return predict_score

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        ablation_set = encoder_config["ablation_set"]
        batch_size, seq_len = batch["question_seq"].size(0), batch["question_seq"].size(1)

        question_emb = self.embed_question(batch["question_seq"])
        interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
        correct_emb = batch["correct_seq"].view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        if ablation_set == 0:
            use_time_seq = batch["use_time_seq"]
            use_time_emb = self.embed_answer_time(use_time_seq)
            learning_emb = self.linear_1(torch.cat((question_emb, use_time_emb, correct_emb), 2))
        else:
            learning_emb = self.linear_1(torch.cat((question_emb, correct_emb), 2))

        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_k)).to(self.params["device"])
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])
        pred = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            it = interval_time_emb[:, t]
            learning = learning_emb[:, t]

            # Learning Module
            learning_gain = self.tanh(self.linear_2(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            LG = gamma_l * ((learning_gain + 1) / 2)

            # Forgetting Module
            gamma_f = self.sig(self.linear_4(torch.cat((h_pre, LG, it), dim=1)))
            h = self.dropout(LG) + gamma_f * h_pre

            # Predicting Module
            y = self.predict_score(h, question_emb[:, t + 1], batch["question_seq"][:, t+1])
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h

        return pred

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss


class LPKTPlus(nn.Module):
    model_name = "LPKTPlus"

    def __init__(self, params, objects):
        super(LPKTPlus, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        num_concept = encoder_config["num_concept"]
        num_question = encoder_config["num_question"]
        num_interval_time = encoder_config["num_interval_time"]
        num_use_time = encoder_config["num_use_time"]
        dim_correct = encoder_config["dim_correct"]
        dim_e = encoder_config["dim_e"]
        dim_k = encoder_config["dim_k"]
        dropout = encoder_config["dropout"]
        ablation_set = encoder_config["ablation_set"]

        self.embed_question = nn.Embedding(num_question + 1, dim_k)
        if ablation_set == 0:
            self.embed_answer_time = nn.Embedding(num_use_time + 1, dim_k)
            self.embed_interval_time = nn.Embedding(num_interval_time + 1, dim_k)
            self.linear_1 = nn.Linear(dim_correct + dim_e + dim_k, dim_k)
        elif ablation_set == 1:
            self.embed_interval_time = nn.Embedding(num_interval_time + 1, dim_k)
            self.linear_1 = nn.Linear(dim_correct + dim_e, dim_k)
        else:
            raise NotImplementedError()
        self.linear_2 = nn.Linear(4 * dim_k, dim_k)
        self.linear_3 = nn.Linear(4 * dim_k, dim_k)
        self.linear_4 = nn.Linear(3 * dim_k, dim_k)
        self.proj_latent2ability = nn.Linear(dim_k, num_concept)
        self.proj_que2difficulty = nn.Linear(dim_k, num_concept)
        self.proj_que2discrimination = nn.Linear(dim_k, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        ablation_set = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]["ablation_set"]
        # use_init_weight = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]["use_init_weight"]

        if ablation_set == 0:
            torch.nn.init.xavier_uniform_(self.embed_answer_time.weight)
        if ablation_set == 0 or ablation_set == 1:
            torch.nn.init.xavier_uniform_(self.embed_interval_time.weight)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)

        # if use_init_weight:
        #     self.proj_latent2ability.weight = nn.Parameter(self.objects["lpkt_plus"]["user_proj_weight_init_value"])
        #     torch.nn.init.constant_(self.proj_latent2ability.bias, 0)
        # else:
        #     torch.nn.init.xavier_uniform_(self.proj_latent2ability.weight)
        torch.nn.init.xavier_uniform_(self.proj_latent2ability.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2difficulty.weight)
        torch.nn.init.xavier_uniform_(self.proj_que2discrimination.weight)

    # ------------------------------------------------------base--------------------------------------------------------
    def predict_score(self, latent, question_emb):
        user_ability = torch.sigmoid(self.proj_latent2ability(self.dropout(latent)))
        que_difficulty = torch.sigmoid(self.proj_que2difficulty(self.dropout(question_emb)))
        que_discrimination = torch.sigmoid(self.proj_que2discrimination(self.dropout(question_emb))) * 10
        predict_score = torch.sigmoid(
            torch.sum(que_discrimination * (user_ability - que_difficulty) * que_difficulty, dim=-1)
        )

        return predict_score

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        ablation_set = encoder_config["ablation_set"]
        batch_size, seq_len = batch["question_seq"].size(0), batch["question_seq"].size(1)

        question_emb = self.embed_question(batch["question_seq"])
        interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
        correct_emb = batch["correct_seq"].view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        if ablation_set == 0:
            use_time_seq = batch["use_time_seq"]
            use_time_emb = self.embed_answer_time(use_time_seq)
            learning_emb = self.linear_1(torch.cat((question_emb, use_time_emb, correct_emb), 2))
        else:
            learning_emb = self.linear_1(torch.cat((question_emb, correct_emb), 2))

        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_k)).to(self.params["device"])
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])
        predict_score = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            it = interval_time_emb[:, t]
            learning = learning_emb[:, t]

            # Learning Module
            learning_gain = self.tanh(self.linear_2(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            LG = gamma_l * ((learning_gain + 1) / 2)

            # Forgetting Module
            gamma_f = self.sig(self.linear_4(torch.cat((h_pre, LG, it), dim=1)))
            h = LG + gamma_f * h_pre

            # Predicting Module
            y = self.predict_score(h, question_emb[:, t + 1])
            predict_score[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h

        return predict_score

    def get_que_diff_pred_loss(self, target_question):
        Q_table_mask = self.objects["cognition_tracing"]["Q_table_mask"]
        que_diff_label = self.objects["cognition_tracing"]["que_diff_ground_truth"]
        mask = Q_table_mask[target_question].bool()

        question_emb = self.embed_question(target_question)
        pred_diff_all = torch.sigmoid(self.proj_que2difficulty(question_emb))
        pred_diff = torch.masked_select(pred_diff_all, mask)
        ground_truth = torch.masked_select(que_diff_label[target_question], mask)
        predict_loss = nn.functional.mse_loss(pred_diff, ground_truth)

        return predict_loss

    def get_que_disc_pred_loss(self, target_question):
        question_emb = self.embed_question(target_question)
        pred_disc = torch.sigmoid(self.proj_que2discrimination(question_emb)).squeeze(dim=-1) * 10
        ground_truth = self.objects["cognition_tracing"]["que_disc_ground_truth"]
        predict_loss = nn.functional.mse_loss(pred_disc, ground_truth)

        return predict_loss

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT_PLUS"]
        num_concept = encoder_config["num_concept"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        ablation_set = encoder_config["ablation_set"]
        # use_init_weight = encoder_config["use_init_weight"]
        w_penalty_neg = self.params["loss_config"]["penalty neg loss"]
        w_learning = self.params["loss_config"]["learning loss"]
        batch_size, seq_len = batch["question_seq"].size(0), batch["question_seq"].size(1)

        question_emb = self.embed_question(batch["question_seq"])
        interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
        correct_emb = batch["correct_seq"].view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        # cf: counterfactual
        cf_correct_emb = (1 - batch["correct_seq"]).view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        if ablation_set == 0:
            use_time_seq = batch["use_time_seq"]
            use_time_emb = self.embed_answer_time(use_time_seq)
            learning_emb = self.linear_1(torch.cat((question_emb, use_time_emb, correct_emb), 2))
            cf_learning_emb = self.linear_1(torch.cat((question_emb, use_time_emb, cf_correct_emb), 2))
        else:
            learning_emb = self.linear_1(torch.cat((question_emb, correct_emb), 2))
            cf_learning_emb = self.linear_1(torch.cat((question_emb, cf_correct_emb), 2))

        # if use_init_weight:
        #     h_pre = torch.ones(batch_size, dim_k).to(self.params["device"])
        # else:
        #     h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_k)).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_k)).to(self.params["device"])
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])
        predict_score_all = torch.zeros(batch_size, seq_len).to(self.params["device"])
        inter_func_in_all = torch.zeros(batch_size, seq_len, num_concept).to(self.params["device"])
        user_ability_all = torch.zeros(batch_size, seq_len, num_concept).to(self.params["device"])
        user_ability_all[:, 0] = torch.sigmoid(self.proj_latent2ability(h_pre))
        cf_user_ability_all = torch.zeros(batch_size, seq_len, num_concept).to(self.params["device"])

        for t in range(0, seq_len - 1):
            it = interval_time_emb[:, t]
            learning = learning_emb[:, t]

            # Learning Module
            learning_gain = self.tanh(self.linear_2(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_pre), dim=1)))
            LG = gamma_l * ((learning_gain + 1) / 2)

            # Forgetting Module
            gamma_f = self.sig(self.linear_4(torch.cat((h_pre, LG, it), dim=1)))
            h = LG + gamma_f * h_pre

            # Predicting Module
            user_ability = torch.sigmoid(self.proj_latent2ability(self.dropout(h)))
            que_difficulty = torch.sigmoid(self.proj_que2difficulty(self.dropout(question_emb[:, t + 1])))
            que_discrimination = torch.sigmoid(self.proj_que2discrimination(self.dropout(question_emb[:, t + 1]))) * 10

            if w_learning != 0:
                user_ability_all[:, t + 1] = user_ability
                cf_learning = cf_learning_emb[:, t]

                cf_learning_gain = self.tanh(self.linear_2(torch.cat((learning_pre, it, cf_learning, h_pre), dim=1)))
                cf_gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, cf_learning, h_pre), dim=1)))
                cf_LG = cf_gamma_l * ((cf_learning_gain + 1) / 2)
                cf_gamma_f = self.sig(self.linear_4(torch.cat((h_pre, cf_LG, it), dim=1)))
                cf_h = cf_LG + cf_gamma_f * h_pre
                cf_user_ability = torch.sigmoid(self.proj_latent2ability(self.dropout(cf_h)))
                cf_user_ability_all[:, t + 1] = cf_user_ability

            interaction_func_in = user_ability - que_difficulty
            inter_func_in_all[:, t + 1] = interaction_func_in
            predict_score = torch.sigmoid(
                torch.sum(que_discrimination * (user_ability - que_difficulty) * que_difficulty, dim=-1)
            )
            predict_score_all[:, t + 1] = predict_score

            # prepare for next prediction
            learning_pre = learning
            h_pre = h

        loss = 0.
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_score_all = torch.masked_select(predict_score_all[:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score_all.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        # 对于做对的题，惩罚user_ability - que_difficulty小于0的值（只惩罚考察的知识点）
        q2c_table = self.objects["data"]["q2c_table"][batch["question_seq"][:, 1:]]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"][batch["question_seq"][:, 1:]]
        if w_penalty_neg != 0:
            target_inter_func_in = torch.gather(inter_func_in_all[:, 1:], 2, q2c_table)
            mask4inter_func_in = mask_bool_seq[:, 1:].unsqueeze(-1) & \
                                 batch["correct_seq"][:, 1:].bool().unsqueeze(-1) & \
                                 q2c_mask_table.bool()
            target_inter_func_in = torch.masked_select(target_inter_func_in, mask4inter_func_in)
            neg_inter_func_in = target_inter_func_in[target_inter_func_in <= 0]
            if neg_inter_func_in.numel() > 0:
                penalty_neg_loss = -neg_inter_func_in.mean()
                if loss_record is not None:
                    num_sample = neg_inter_func_in.shape[0]
                    loss_record.add_loss("penalty neg loss", penalty_neg_loss.detach().cpu().item() * num_sample, num_sample)
                loss = loss + penalty_neg_loss * w_penalty_neg

        # 反事实约束：做对一道题比做错一道题的增长要大；做了题要比不做题增长大
        if w_learning != 0:
            master_leval = user_ability_all[:, 2:] - user_ability_all[:, 1:-1]
            mask4master = mask_bool_seq[:, 1:-1].unsqueeze(-1) & q2c_mask_table[:, :-1].bool()

            learn_loss = 0.
            # 对于每个时刻，都应该比上个时刻有所增长（对应当前时刻所做题的知识点），惩罚小于0的部分
            target_neg_master_leval = torch.masked_select(torch.gather(master_leval, 2, q2c_table[:, :-1]), mask4master)
            neg_master_leval = target_neg_master_leval[target_neg_master_leval < 0]
            num_sample = neg_master_leval.numel()
            if num_sample > 0:
                learn_loss1 = -neg_master_leval.mean()
                learn_loss = learn_loss + learn_loss1
                if loss_record is not None:
                    loss_record.add_loss("learning loss", learn_loss1.detach().cpu().item() * num_sample, num_sample)

            f_minus_cf = torch.gather(user_ability_all[:, 1:] - cf_user_ability_all[:, 1:], 2, q2c_table)
            correct_seq1 = batch["correct_seq"][:, :-1].bool()
            correct_seq2 = (1 - batch["correct_seq"][:, :-1]).bool()
            mask4correct = mask_bool_seq[:, 1:].unsqueeze(-1) & correct_seq1.unsqueeze(-1) & q2c_mask_table.bool()
            mask4wrong = mask_bool_seq[:, 1:].unsqueeze(-1) & correct_seq2.unsqueeze(-1) & q2c_mask_table.bool()

            # 对于做对的时刻，f_minus_cf应该大于0，惩罚小于0的部分
            target_neg_f_minus_cf = torch.masked_select(f_minus_cf, mask4correct)
            neg_f_minus_cf = target_neg_f_minus_cf[target_neg_f_minus_cf < 0]
            num_sample = neg_f_minus_cf.numel()
            if num_sample > 0:
                learn_loss2 = -neg_f_minus_cf.mean()
                learn_loss = learn_loss + learn_loss2
                if loss_record is not None:
                    loss_record.add_loss("learning loss", learn_loss2.detach().cpu().item() * num_sample, num_sample)

            # 对于做错的时刻，f_minus_cf应该小于0，惩罚大于0的部分
            target_pos_f_minus_cf = torch.masked_select(f_minus_cf, mask4wrong)
            pos_f_minus_cf = target_pos_f_minus_cf[target_pos_f_minus_cf > 0]
            num_sample = pos_f_minus_cf.numel()
            if num_sample > 0:
                learn_loss3 = pos_f_minus_cf.mean()
                learn_loss = learn_loss + learn_loss3
                if loss_record is not None:
                    loss_record.add_loss("learning loss", learn_loss3.detach().cpu().item() * num_sample, num_sample)

            loss = loss + learn_loss * w_learning

        return loss
