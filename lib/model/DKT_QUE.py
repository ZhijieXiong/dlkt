import torch
import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer2
from .Module.PredictorLayer import PredictorLayer
from .util import get_mask4last_or_penultimate


class DKT_QUE(nn.Module):
    model_name = "DKT_QUE"
    use_question = True

    def __init__(self, params, objects):
        super(DKT_QUE, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer2(params["models_config"]["kt_model"]["kt_embed_layer"])
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_QUE"]
        dim_emb = encoder_config["dim_emb"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]

        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb * 2, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb * 2, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb * 2, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.encoder_layer = self.encoder_layer

        self.predict_layer = PredictorLayer(self.params, self.objects)

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DKT_QUE"]
        dim_emb = encoder_config["dim_emb"]

        self.encoder_layer.flatten_parameters()
        batch_size = batch["correct_seq"].shape[0]
        correct_emb = batch["correct_seq"].reshape(-1, 1).repeat(1, dim_emb).reshape(batch_size, -1, dim_emb)
        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        interaction_emb = torch.cat((question_emb[:, :-1], correct_emb[:, :-1]), dim=2)
        latent, _ = self.encoder_layer(interaction_emb)
        predict_layer_input = torch.cat((latent, question_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score_result = self.get_predict_score(batch)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)[:, :-1]
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
        predict_score = predict_score * mask4last
        predict_score = torch.sum(predict_score, dim=1)

        return predict_score
