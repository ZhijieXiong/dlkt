from .util import *
from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from ..CONSTANT import HAS_TIME, HAS_USE_TIME, HAS_NUM_HINT, HAS_NUM_ATTEMPT


# class AuxInfoLPKT(nn.Module):
#     model_name = "AuxInfoLPKT"
#
#     def __init__(self, params, objects):
#         super(AuxInfoLPKT, self).__init__()
#         self.params = params
#         self.objects = objects
#
#         encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoDKT"]
#         dataset_name = encoder_config["dataset_name"]
#         dim_question = encoder_config["dim_question"]
#         dim_latent = encoder_config["dim_latent"]
#         rnn_type = encoder_config["rnn_type"]
#         num_rnn_layer = encoder_config["num_rnn_layer"]
#
#         self.has_time = dataset_name in HAS_TIME
#         self.has_use_time = dataset_name in HAS_USE_TIME
#         self.has_count = dataset_name in HAS_COUNT
#         self.has_use_time_and_count = dataset_name in set(HAS_USE_TIME).intersection(set(HAS_COUNT))
#
#         # embedding层
#         self.embed_layer = KTEmbedLayer(self.params, self.objects)
#
#         # 输入embedding融合层（在前端就处理好，每种辅助信息的toke表不超过100）
#         if self.has_time:
#             self.embed_interval_time = nn.Embedding(100, dim_question)
#         if self.has_use_time:
#             self.embed_use_time = nn.Embedding(100, dim_question)
#         if self.has_count:
#             self.embed_num_hint = nn.Embedding(100, dim_question)
#             self.embed_num_attempt = nn.Embedding(100, dim_question)
#         # 融合h、interval time
#         self.fuse_layer1 = nn.Linear(dim_question * 2, dim_question)
#         # 融合h、use time
#         self.fuse_layer2 = nn.Linear(dim_question * 2, dim_question)
#         # 融合question、concept、correct
#         self.fuse_layer3 = nn.Linear(dim_question * 3, dim_question)
#         # 融合h、use time、num hint、num attempt
#         self.fuse_layer4 = nn.Linear(dim_question * 4, dim_question)
#
#         # encode层：RNN
#         if (self.has_use_time or self.has_count) and self.has_time:
#             dim_rrn_input = dim_question * 3
#         elif (self.has_use_time or self.has_count) or self.has_time:
#             dim_rrn_input = dim_question * 2
#         else:
#             dim_rrn_input = dim_question
#         if rnn_type == "rnn":
#             self.encoder_layer = nn.RNN(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
#         elif rnn_type == "lstm":
#             self.encoder_layer = nn.LSTM(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
#         else:
#             self.encoder_layer = nn.GRU(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
#
#         # 预测层
#         self.predict_layer = PredictorLayer(self.params, self.objects)


class AuxInfoQDKT(nn.Module):
    model_name = "AuxInfoQDKT"

    def __init__(self, params, objects):
        super(AuxInfoQDKT, self).__init__()
        self.params = params
        self.objects = objects

        # embedding层
        self.embed_layer = KTEmbedLayer(self.params, self.objects)

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoQDKT"]
        dataset_name = encoder_config["dataset_name"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        self.has_time = dataset_name in HAS_TIME
        self.has_use_time = dataset_name in HAS_USE_TIME
        self.has_num_hint = dataset_name in HAS_NUM_HINT
        self.has_num_attempt = dataset_name in HAS_NUM_ATTEMPT

        # 输入embedding融合层（在前端就处理好，每种辅助信息的toke表不超过100）
        if self.has_time:
            self.embed_interval_time = nn.Embedding(100, dim_question)
        if self.has_use_time:
            self.embed_use_time = nn.Embedding(100, dim_question)
        if self.has_num_hint:
            self.embed_num_hint = nn.Embedding(100, dim_question)
        if self.has_num_attempt:
            self.embed_num_attempt = nn.Embedding(100, dim_question)
        # 融合question、concept、correct
        self.fuse_q_c_c = nn.Linear(dim_question * 3, dim_question)
        # 融合use time、num hint、num attempt
        self.fuse_ut_nh_na = nn.Linear(dim_question * 3, dim_question)

        # encode层：RNN
        if (self.has_use_time or self.has_num_hint or self.has_num_attempt) and self.has_time:
            dim_rrn_input = dim_question * 3
        elif (self.has_use_time or self.has_num_hint or self.has_num_attempt) or self.has_time:
            dim_rrn_input = dim_question * 2
        else:
            dim_rrn_input = dim_question
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_rrn_input, dim_latent, batch_first=True, num_layers=num_rnn_layer)

        # 预测层
        self.predict_layer = PredictorLayer(self.params, self.objects)

    def get_qc_emb4single_concept(self, batch):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        if (not use_LLM_emb4question) and (not use_LLM_emb4concept):
            concept_question_emb = self.embed_layer.get_emb_concatenated(("concept", "question"),
                                                                         (concept_seq, question_seq))
        else:
            concept_emb = self.embed_layer.get_emb("concept", concept_seq)
            question_emb = self.embed_layer.get_emb("question", question_seq)
            if use_LLM_emb4concept:
                concept_emb = self.MLP4concept(concept_emb)
            if use_LLM_emb4question:
                question_emb = self.MLP4question(question_emb)
            concept_question_emb = torch.cat((concept_emb, question_emb), dim=-1)

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], fusion_type="mean")

    def get_interaction_emb4single_concept(self, batch):
        qc_emb = self.get_qc_emb4single_concept(batch)
        correct_emb = self.embed_layer.get_emb("correct", batch["correct_seq"])
        return torch.cat((qc_emb, correct_emb), dim=-1)

    def forward(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        dim_question = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoQDKT"]["dim_question"]
        weight_aux_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["AuxInfoQDKT"]["weight_aux_emb"]
        batch_size, seq_len = batch["correct_seq"].shape[0], batch["correct_seq"].shape[1]

        correct_emb = self.embed_layer.get_emb("correct", batch["correct_seq"])
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        interaction_emb = self.fuse_q_c_c(torch.cat((qc_emb, correct_emb), dim=-1))

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

        predict_layer_input = torch.cat((latent[:, :-1], qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

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

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)
