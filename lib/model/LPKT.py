from .util import *
from .loss_util import binary_entropy


class LPKT(nn.Module):
    model_name = "LPKT"
    use_question = True

    def __init__(self, params, objects):
        super(LPKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
        num_question = encoder_config["num_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_e = encoder_config["dim_e"]
        dim_k = encoder_config["dim_k"]
        dropout = encoder_config["dropout"]
        ablation_set = encoder_config["ablation_set"]

        # 3600 sec: 1 hour, 43200 min: 1 month
        if (ablation_set == 0) or (ablation_set == 2):
            self.at_embed = nn.Embedding(3600 + 1, dim_k)
            torch.nn.init.xavier_uniform_(self.at_embed.weight)
        if (ablation_set == 0) or (ablation_set == 1):
            self.it_embed = nn.Embedding(43200 + 1, dim_k)
            torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(num_question + 1, dim_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        if ablation_set == 0:
            # 完整版
            dim_in_1 = dim_correct + dim_e + dim_k
            dim_in_2 = 4 * dim_k
            dim_in_3 = 4 * dim_k
            dim_in_4 = 3 * dim_k
        elif ablation_set == 1:
            # 没有use time
            dim_in_1 = dim_correct + dim_e
            dim_in_2 = 4 * dim_k
            dim_in_3 = 4 * dim_k
            dim_in_4 = 3 * dim_k
        elif ablation_set == 2:
            # 没有interval time
            dim_in_1 = dim_correct + dim_e + dim_k
            dim_in_2 = 3 * dim_k
            dim_in_3 = 3 * dim_k
            dim_in_4 = 2 * dim_k
        else:
            # 没有use time和interval time
            dim_in_1 = dim_correct + dim_e
            dim_in_2 = 3 * dim_k
            dim_in_3 = 3 * dim_k
            dim_in_4 = 2 * dim_k

        self.linear_1 = nn.Linear(dim_in_1, dim_k)
        self.linear_2 = nn.Linear(dim_in_2, dim_k)
        self.linear_3 = nn.Linear(dim_in_3, dim_k)
        self.linear_4 = nn.Linear(dim_in_4, dim_k)
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
        correct_seq = batch["correct_seq"]

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
        num_concept = encoder_config["num_concept"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        ablation_set = encoder_config["ablation_set"]
        q_matrix = self.objects["LPKT"]["q_matrix"]

        batch_size, seq_len = question_seq.size(0), question_seq.size(1)
        e_embed_data = self.e_embed(question_seq)

        it_embed_data = None
        if (ablation_set == 0) or (ablation_set == 1):
            interval_time_seq = batch["interval_time_seq"]
            it_embed_data = self.it_embed(interval_time_seq)
        correct_seq = correct_seq.view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        h_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_k)).repeat(batch_size, 1, 1).to(self.params["device"])
        h_tilde_pre = None
        if (ablation_set == 0) or (ablation_set == 2):
            use_time_seq = batch["use_time_seq"]
            at_embed_data = self.at_embed(use_time_seq)
            all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, correct_seq), 2))
        else:
            all_learning = self.linear_1(torch.cat((e_embed_data, correct_seq), 2))
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])

        pred = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            e = question_seq[:, t]
            # q_e: (bs, 1, n_skill)
            q_e = q_matrix[e].view(batch_size, 1, -1)

            it = None
            if (ablation_set == 0) or (ablation_set == 1):
                it = it_embed_data[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, dim_k)
            learning = all_learning[:, t]

            if (ablation_set == 0) or (ablation_set == 1):
                learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            else:
                learning_gain = self.linear_2(torch.cat((learning_pre, learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            if (ablation_set == 0) or (ablation_set == 1):
                gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            else:
                gamma_l = self.linear_3(torch.cat((learning_pre, learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            if (ablation_set == 0) or (ablation_set == 1):
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, dim_k),
                    it.repeat(1, n_skill).view(batch_size, -1, dim_k)
                ), 2)))
            else:
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, dim_k)
                ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = q_matrix[question_seq[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, dim_k)
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / dim_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
