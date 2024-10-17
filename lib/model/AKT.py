import torch
import torch.nn as nn

from .Module.EncoderLayer import EncoderLayer
from .Module.KTEmbedLayer import KTEmbedLayer
from .util import get_mask4last_or_penultimate


class AKT(nn.Module):
    model_name = "AKT"
    use_question = True

    def __init__(self, params, objects):
        super(AKT, self).__init__()
        self.params = params
        self.objects = objects

        # embed init
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        dim_emb = encoder_config["dim_model"]
        separate_qa = encoder_config["separate_qa"]
        num_question = encoder_config["num_question"]
        num_concept = encoder_config["num_concept"]
        self.embed_question_difficulty = nn.Embedding(num_question, 1)
        self.embed_concept_variation = nn.Embedding(num_concept, dim_emb)
        self.embed_interaction_variation = nn.Embedding(2 * num_concept, dim_emb)

        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        if separate_qa:
            self.embed_interaction = nn.Embedding(2 * num_concept + 1, dim_emb)
        else:
            self.embed_interaction = nn.Embedding(2, dim_emb)

        self.encoder_layer = EncoderLayer(params, objects)

        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dropout = encoder_config["dropout"]
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

        # 对性能来说至关重要的一步
        for p in self.parameters():
            if p.size(0) == num_question and num_question > 0:
                torch.nn.init.constant_(p, 0.)

    def get_concept_emb_all(self):
        return self.embed_concept.weight

    def get_concept_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept(batch["concept_seq"])

        return concept_emb

    def get_concept_variation_emb(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept_variation,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
        else:
            concept_emb = self.embed_concept_variation(batch["concept_seq"])

        return concept_emb

    def base_emb(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        correct_seq = batch["correct_seq"]

        # c_ct
        concept_emb = self.get_concept_emb(batch)
        if separate_qa:
            # todo: 有问题，如果是only question也要融合interaction_emb
            concept_seq = batch["concept_seq"]
            interaction_seq = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb
        return concept_emb, interaction_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["AKT"]
        separate_qa = encoder_config["separate_qa"]
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        concept_variation_emb = self.get_concept_variation_emb(batch)
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        if separate_qa:
            # uq * f_(ct,rt) + e_(ct,rt)
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)
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
        question_difficulty_emb = self.embed_question_difficulty(batch["question_seq"])
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
