import torch
import torch.nn as nn

from .Module.EncoderLayer import EncoderLayer
from .Module.KTEmbedLayer import KTEmbedLayer2
from .util import get_mask4last_or_penultimate


class AKT_QUE(nn.Module):
    model_name = "AKT_QUE"
    use_question = True

    def __init__(self, params, objects):
        super(AKT_QUE, self).__init__()
        self.params = params
        self.objects = objects

        # embed init
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dropout = encoder_config["dropout"]

        self.embed_layer = KTEmbedLayer2(params["models_config"]["kt_model"]["kt_embed_layer"])
        self.encoder_layer = EncoderLayer(params, objects)
        self.predict_layer = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def base_emb(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        num_question = encoder_config["num_question"]
        separate_qa = encoder_config["separate_qa"]
        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        if separate_qa:
            interaction_seq = batch["question_seq"] + num_question * batch["correct_seq"]
            interaction_emb = self.embed_layer.get_emb("interaction", interaction_seq)
        else:
            interaction_emb = self.embed_layer.get_emb("interaction", batch["correct_seq"]) + question_emb
        return question_emb, interaction_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        question_emb, interaction_emb = self.base_emb(batch)
        question_variation_emb = self.embed_layer.get_emb("question_variation", question_seq)
        question_difficulty_emb = self.embed_layer.get_emb("question_difficulty", question_seq)
        question_emb = question_emb + question_difficulty_emb * question_variation_emb
        interaction_variation_emb = self.embed_layer.get_emb("interaction_variation", correct_seq)
        if separate_qa:
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + question_variation_emb)
        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }
        latent = self.encoder_layer(encoder_input)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

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

        # 判断是否有样本损失权重设置
        sample_weight = None
        if self.params.get("sample_reweight", False) and self.params["sample_reweight"].get("use_sample_reweight", False):
            sample_weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])

        # 计算损失
        predict_loss = nn.functional.binary_cross_entropy(
            predict_score.double(), ground_truth.double(), weight=sample_weight
        )
        question_difficulty_emb = self.embed_layer.get_emb("question_difficulty", batch["question_seq"])
        rasch_loss = (question_difficulty_emb ** 2.).sum()
        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch loss"]

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "rasch loss": {
                    "value": rasch_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)

        return predict_score[mask4last.bool()]
