import torch
import torch.nn as nn

from ..CONSTANT import HAS_TIME, HAS_USE_TIME


def get_mirt_predict_score(user_ability, question_diff, question_disc, q2c_mask):
    irt_logits = question_disc * (torch.sum((user_ability - question_diff) * q2c_mask, dim=-1, keepdim=True))
    irt_output = torch.sigmoid(irt_logits).squeeze(1)
    return irt_output


class NeuroMIRT(nn.Module):
    model_name = "NeuroMIRT"

    def __init__(self, params, objects):
        super(NeuroMIRT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["NeuroMIRT"]
        dataset_name = encoder_config["dataset_name"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        num_concept_combination = encoder_config["num_concept_combination"]
        dim_emb = encoder_config["dim_emb"]
        dropout = encoder_config["dropout"]

        self.has_time = dataset_name in HAS_TIME
        self.has_use_time = dataset_name in HAS_USE_TIME

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_question_disc = nn.Embedding(num_question, 1)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_concept_guess = nn.Embedding(num_concept_combination, 1)

        self.embed_interval_time = nn.Embedding(100, dim_emb)
        self.embed_use_time = nn.Embedding(100, dim_emb)
        self.embed_hint_attempt = nn.Embedding(12, dim_emb)

        self.fuse_qa = nn.Linear(dim_emb * 2, dim_emb)
        self.absorb = nn.Linear(dim_emb * 3, dim_emb)
        self.forget = nn.Linear(dim_emb * 3, dim_emb)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["NeuroMIRT"]["dim_emb"]
        q2c_table = self.objects["data"]["q2c_table"]
        q2c_mask_table = self.objects["data"]["q2c_mask_table"]

        question_seq = batch["question_seq"]
        hint_attempt_seq = batch["hint_attempt_seq"]
        correct_seq = batch["correct_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        if self.has_time:
            interval_time_emb = self.embed_interval_time(batch["interval_time_seq"])
        else:
            interval_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        if self.has_use_time:
            use_time_emb = self.embed_use_time(batch["use_time_seq"])
        else:
            use_time_emb = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        hint_attempt_emb = self.embed_hint_attempt(hint_attempt_seq)
        question_emb = self.embed_question(question_seq)
        question_disc = torch.sigmoid(self.embed_question_disc(question_seq)) * 10
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        # concept_prompt: (bs, seq_len, num_max_c, dim_emb)
        concept_prompt = self.embed_concept(q2c_table[question_seq])
        q2c_mask = q2c_mask_table[question_seq]

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        latent_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            interaction_emb = self.fuse_qa(torch.cat((question_emb[:, t], correct_emb[:, t]), dim=1))
            absorb = self.absorb(torch.cat((interaction_emb, use_time_emb[:, t], hint_attempt_emb[:, t]), dim=1))
            forget = torch.sigmoid(self.forget(torch.cat((latent_pre, interval_time_emb[:, t], interaction_emb), dim=1)))
            latent_current = latent_pre * forget + self.dropout(absorb)
            # question emb和concept prompt计算相似度并放缩到0~1，得到习题难度
            question_diff = (torch.cosine_similarity(question_emb[:, t].unsqueeze(1), concept_prompt[:, t], dim=-1) + 1) / 2
            # 类似question_diff，计算出学生能力
            user_ability = (torch.cosine_similarity(latent_current.unsqueeze(1), concept_prompt[:, t+1], dim=-1) + 1) / 2
            y_current = get_mirt_predict_score(user_ability, question_diff, question_disc[:, t+1], q2c_mask[:, t])
            y[:, t] = y_current
            latent[:, t + 1, :] = latent_current
            latent_pre = latent_current

        return y

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, :-1], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, 1:]

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss
