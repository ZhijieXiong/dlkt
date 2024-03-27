import torch.nn.init

from .util import *
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
        que_user_share_proj = encoder_config["que_user_share_proj"]
        num_mlp_layer = encoder_config["num_mlp_layer"]
        dropout = encoder_config["dropout"]

        self.has_time = dataset_name in HAS_TIME
        self.has_use_time = dataset_name in HAS_USE_TIME
        self.has_num_hint = dataset_name in HAS_NUM_HINT
        self.has_num_attempt = dataset_name in HAS_NUM_ATTEMPT

        # 输入embedding融合层（在前端就处理好，每种辅助信息的toke表不超过100）
        self.embed_question = nn.Embedding(num_question, dim_question)
        torch.nn.init.xavier_uniform_(self.embed_question.weight)
        if self.has_time:
            self.embed_interval_time = nn.Embedding(100, dim_question)
        if self.has_use_time:
            self.embed_use_time = nn.Embedding(100, dim_question)
        if self.has_num_hint:
            self.embed_num_hint = nn.Embedding(100, dim_question)
        if self.has_num_attempt:
            self.embed_num_attempt = nn.Embedding(100, dim_question)
        # 融合use time、num hint、num attempt
        self.fuse_ut_nh_na = nn.Linear(dim_question * 3, dim_question)

        dim_rnn_output = dim_question if que_user_share_proj else dim_latent
        # encode层：RNN
        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * 4
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * 3
        else:
            dim_rrn_input = dim_question * 2
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

    def get_question_diff(self, batch_question):
        test_theory = self.params["other"]["cognition_tracing"]["test_theory"]
        question_emb = self.embed_question(batch_question)
        if test_theory == "rasch":
            que_difficulty = self.que2difficulty(self.dropout(question_emb))
        else:
            que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))

        return que_difficulty

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
            y = (que_discrimination * (user_ability - que_difficulty)) * \
                que_difficulty / torch.sum(que_difficulty, dim=1, keepdim=True)

            # que_difficulty = torch.sigmoid(self.que2difficulty(self.dropout(question_emb)))
            # que_diff_mask = torch.ones_like(que_difficulty).float().to(self.params["device"])
            # que_diff_mask[que_difficulty < 0.05] = 0
            # y = (que_discrimination * (user_ability - que_difficulty)) * \
            #     que_difficulty * que_diff_mask / torch.sum(que_difficulty, dim=1, keepdim=True)
        predict_score = torch.sigmoid(torch.sum(y, dim=-1))

        return predict_score

    def forward(self, batch):
        dim_question = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["dim_question"]
        weight_aux_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDCT"]["weight_aux_emb"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_question).reshape(batch_size, -1, dim_question)
        question_emb = self.embed_question(question_seq)
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
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss
