import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .util import get_mask4last_or_penultimate


class DKT(nn.Module):
    model_name = "DKT"

    def __init__(self, params, objects):
        super(DKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        dim_emb = encoder_config["dim_emb"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.encoder_layer = self.encoder_layer

        self.predict_layer = PredictorLayer(self.params, self.objects)

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT"]
        use_concept = encoder_config["use_concept"]
        if use_concept:
            pass
        else:
            pass
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        dim_predict_out = self.params["models_config"]["kt_model"]["predict_layer"]["direct"]["dim_predict_out"]
        interaction_seq = concept_seq + dim_predict_out * correct_seq
        interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        predict_score = self.predict_layer(latent)

        return predict_score

    def get_latent(self, batch):
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        dim_predict_out = self.params["models_config"]["kt_model"]["predict_layer"]["direct"]["dim_predict_out"]
        interaction_seq = concept_seq + dim_predict_out * correct_seq
        interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        return latent

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        dim_predict_out = self.params["models_config"]["kt_model"]["predict_layer"]["direct"]["dim_predict_out"]
        one_hot4predict_score = nn.functional.one_hot(batch["concept_seq"][:, 1:], dim_predict_out)
        predict_score = self.forward(batch)[:, :-1]
        predict_score = (predict_score * one_hot4predict_score).sum(-1)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        dim_predict_out = self.params["models_config"]["kt_model"]["predict_layer"]["direct"]["dim_predict_out"]
        one_hot4predict_score = nn.functional.one_hot(batch["concept_seq"][:, 1:], dim_predict_out)
        predict_score = self.forward(batch)[:, :-1]
        predict_score = (predict_score * one_hot4predict_score).sum(-1)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
        predict_score = predict_score * mask4last
        predict_score = torch.sum(predict_score, dim=1)

        return predict_score
