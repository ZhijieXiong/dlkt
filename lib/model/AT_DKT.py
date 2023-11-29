import torch

from torch import nn

from .Module.KTEmbedLayer import KTEmbedLayer


def ut_mask(seq_len):
    """
    Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)


class AT_DKT(nn.Module):
    def __init__(self, params, objects):
        super(AT_DKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        dropout = encoder_config["dropout"]

        if rnn_type == "rnn":
            self.dkt_encoder = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.dkt_encoder = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.dkt_encoder = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.dkt_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_latent, dim_latent // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_latent // 2, num_concept),
            nn.Sigmoid()
        )

        QT_net_type = encoder_config["QT_net_type"]
        QT_rnn_type = encoder_config["QT_rnn_type"]
        QT_num_rnn_layer = encoder_config["QT_num_rnn_layer"]
        QT_transformer_num_block = encoder_config["QT_transformer_num_block"]
        QT_transformer_num_head = encoder_config["QT_transformer_num_head"]
        if QT_net_type == "rnn":
            if QT_rnn_type == "rnn":
                self.QT_rnn = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=QT_num_rnn_layer)
            elif QT_rnn_type == "lstm":
                self.QT_rnn = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=QT_num_rnn_layer)
            else:
                self.QT_rnn = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=QT_num_rnn_layer)
        elif QT_net_type == "transformer":
            QT_encoder_layer = nn.TransformerEncoderLayer(dim_emb, nhead=QT_transformer_num_head)
            QT_encoder_norm = nn.LayerNorm(dim_emb)
            self.QT_transformer = (
                nn.TransformerEncoder(QT_encoder_layer, num_layers=QT_transformer_num_block, norm=QT_encoder_norm))
        else:
            raise NotImplementedError()
        self.QT_classifier = nn.Sequential(
            nn.Linear(dim_latent, dim_latent // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_latent // 2, num_concept)
        )
        self.IK_predictor = nn.Sequential(
            nn.Linear(dim_latent, dim_latent // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_latent // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        QT_net_type = encoder_config["QT_net_type"]
        num_concept = encoder_config["num_concept"]

        question_seq = batch["question_seq"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        interaction_seq = concept_seq + num_concept * correct_seq

        interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
        question_emb = self.embed_layer.get_emb("question", question_seq)
        concept_emb = self.embed_layer.get_emb("concept", concept_seq)
        cate_emb = question_emb + concept_emb
        seq_len = cate_emb.shape[1]

        # predict concept corresponding to question
        if QT_net_type == "rnn":
            qh, _ = self.QT_rnn(cate_emb)
        elif QT_net_type == "transformer":
            mask = ut_mask(seq_len).to(self.params["device"])
            qh = self.QT_transformer(cate_emb.transpose(0, 1), mask).transpose(0, 1)
        else:
            raise NotImplementedError()
        QT_predict_score = self.QT_classifier(qh)

        # predict right or wrong
        interaction_emb = interaction_emb + qh + concept_emb + question_emb
        latent, _ = self.dkt_encoder(interaction_emb)
        KT_predict_score = self.dkt_classifier(latent)

        # predict student's history accuracy
        IK_predict_score = self.IK_predictor(latent).squeeze(-1)

        return KT_predict_score, QT_predict_score, IK_predict_score

    def get_latent(self, batch):
        pass

    def get_predict_score(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        num_concept = encoder_config["num_concept"]
        concept_seq = batch["concept_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        KT_predict_score, QT_predict_score, IK_predict_score = self.forward(batch)
        KT_predict_score = (KT_predict_score[:, :-1] * nn.functional.one_hot(concept_seq[:, 1:], num_concept)).sum(-1)

        return KT_predict_score[mask_bool_seq[:, 1:]]

    def get_predict_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        num_concept = encoder_config["num_concept"]
        IK_start = encoder_config["IK_start"]
        concept_seq = batch["concept_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        KT_predict_score, QT_predict_score, IK_predict_score = self.forward(batch)
        KT_predict_score = (KT_predict_score[:, :-1] * nn.functional.one_hot(concept_seq[:, 1:], num_concept)).sum(-1)

        KT_predict_loss = nn.functional.binary_cross_entropy(
            KT_predict_score[mask_bool_seq[:, 1:]].double(),
            ground_truth.double()
        )
        QT_predict_loss = nn.functional.cross_entropy(
            QT_predict_score[:, :-1][mask_bool_seq[:, :-1]],
            concept_seq[:, :-1][mask_bool_seq[:, :-1]]
        )
        IK_predict_loss = nn.functional.mse_loss(
            IK_predict_score[:, IK_start+1:][mask_bool_seq[:, IK_start+1:]],
            batch["history_acc_seq"][:, IK_start+1:][mask_bool_seq[:, IK_start+1:]]
        )

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            num_sample_IK = torch.sum(batch["mask_seq"][:, IK_start + 1:]).item()
            loss_record.add_loss("predict loss", KT_predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("QT_loss", QT_predict_loss.detach().cpu().item() * num_sample, num_sample)
            loss_record.add_loss("IK_loss", IK_predict_loss.detach().cpu().item() * num_sample_IK, num_sample_IK)

        return (KT_predict_loss +
                QT_predict_loss * self.params["loss_config"]["QT_loss"] +
                IK_predict_loss * self.params["loss_config"]["IK_loss"])
