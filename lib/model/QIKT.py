import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer


class MLP(nn.Module):
    def __init__(self, num_layer, dim_in, dim_out, dropout):
        super().__init__()

        self.linear_list = nn.ModuleList([
            nn.Linear(dim_in, dim_in)
            for _ in range(num_layer)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(dim_in, dim_out)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.linear_list:
            x = torch.relu(lin(x))
        return self.out(self.dropout(x))


def sigmoid_inverse(x, epsilon=1e-8):
    return torch.log(x / (1 - x + epsilon) + epsilon)


class QIKT(nn.Module):
    model_name = "QIKT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            self.embed_layer.parse_Q_table()

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["QIKT"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        num_mlp_layer = encoder_config["num_mlp_layer"]
        dropout = encoder_config["dropout"]

        if rnn_type == "rnn":
            self.rnn_layer4question = nn.RNN(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.RNN(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.rnn_layer4question = nn.LSTM(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.LSTM(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.rnn_layer4question = nn.GRU(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.GRU(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        self.dropout_layer = nn.Dropout(dropout)
        self.predict_layer4q_next = MLP(num_mlp_layer, dim_emb * 3, 1, dropout)
        self.predict_layer4q_all = MLP(num_mlp_layer, dim_emb, num_question, dropout)
        self.predict_layer4c_next = MLP(num_mlp_layer, dim_emb * 3, num_concept, dropout)
        self.predict_layer4c_all = MLP(num_mlp_layer, dim_emb, num_concept, dropout)
        self.que_discrimination_layer = MLP(num_mlp_layer, dim_emb * 2, 1, dropout)

    def get_concept_emb(self):
        return self.embed_layer.get_emb_all("concept")

    def get_qc_emb4single_concept(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        concept_question_emb = self.embed_layer.get_emb_concatenated(("concept", "question"),
                                                                     (concept_seq, question_seq))

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], concept_fusion="mean")

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        lambda_q_all = encoder_config["lambda_q_all"]
        lambda_c_next = encoder_config["lambda_c_next"]
        lambda_c_all = encoder_config["lambda_c_all"]
        use_irt = encoder_config["use_irt"]
        dim_emb = encoder_config["dim_emb"]
        correct_seq = batch["correct_seq"]
        data_type = self.params["datasets_config"]["data_type"]

        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
            concept_emb = self.embed_layer.get_concept_fused_emb(batch["question_seq"], "mean")
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
            concept_emb = self.embed_layer.get_emb("concept", batch["concept_seq"])
        qca_emb = torch.cat([
            qc_emb.mul((1 - correct_seq).unsqueeze(-1).repeat(1, 1, dim_emb * 2)),
            qc_emb.mul(correct_seq.unsqueeze(-1).repeat(1, 1, dim_emb * 2))
        ],
            dim=-1
        )
        ca_emb = torch.cat([
            concept_emb.mul((1 - correct_seq).unsqueeze(-1).repeat(1, 1, dim_emb)),
            concept_emb.mul(correct_seq.unsqueeze(-1).repeat(1, 1, dim_emb))
        ],
            dim=-1
        )

        latent_question = self.dropout_layer(self.rnn_layer4question(qca_emb[:, :-1])[0])
        latent_concept = self.dropout_layer(self.rnn_layer4concept(ca_emb[:, :-1])[0])

        predict_score_q_next = torch.sigmoid(self.predict_layer4q_next(
            torch.cat((qc_emb[:, 1:], latent_question))
        ))
        predict_score_q_all = torch.sigmoid(self.predict_layer4q_all(latent_question))
        predict_score_c_next = torch.sigmoid(self.predict_layer4c_next(
            torch.cat((concept_emb[:, 1:], latent_concept))
        ))
        predict_score_c_all = torch.sigmoid(self.predict_layer4c_all(latent_concept))

        if use_irt:
            predict_score = (sigmoid_inverse(predict_score_q_all) * lambda_q_all +
                             sigmoid_inverse(predict_score_c_all) * lambda_c_all +
                             sigmoid_inverse(predict_score_c_next) * lambda_c_next)
            predict_score = torch.sigmoid(predict_score)
        else:
            predict_score = (predict_score_q_all * lambda_q_all +
                             predict_score_c_all * lambda_c_all +
                             predict_score_c_next * lambda_c_next)
            predict_score = predict_score / (lambda_q_all + lambda_c_all + lambda_c_next)

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
