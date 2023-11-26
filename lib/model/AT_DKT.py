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

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_interaction = nn.Embedding(num_concept * 2, dim_emb)

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
            nn.Linear(dim_latent, dim_latent // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_latent // 2, num_concept)
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
            nn.Linear(dim_latent // 2, 1)
        )

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AT_DKT"]
        QT_net_type = encoder_config["QT_net_type"]
        num_concept = encoder_config["num_concept"]

        question_seq = batch["question_seq"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        interaction_seq = concept_seq[:, 0:-1] + num_concept * correct_seq[:, 0:-1]

        interaction_emb = self.embed_interaction(interaction_seq)
        question_emb = self.embed_question(question_seq)
        concept_emb = self.embed_concept(concept_seq)
        cate_emb = interaction_emb + question_emb + concept_emb
        seq_len = cate_emb.shape[1]

        # predict concept corresponding to question
        if QT_net_type == "rnn":
            qh, _ = self.QT_rnn(cate_emb)
        elif QT_net_type == "transformer":
            mask = ut_mask(seq_len)
            qh = self.QT_transformer(cate_emb.transpose(0, 1), mask).transpose(0, 1)
        else:
            raise NotImplementedError()
        concept_predict_score = self.QT_classifier(qh)

        # predict student's history accuracy





