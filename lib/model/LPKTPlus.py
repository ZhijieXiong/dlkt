from torch import nn

from .util import *


class LPKTPlus(nn.Module):
    model_name = "LPKTPlus"

    def __init__(self, params, objects):
        super(LPKTPlus, self).__init__()
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
        q_matrix = self.objects["LPKT_PLUS"]["q_matrix"]

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
