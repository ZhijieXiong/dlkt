import torch
import torch.nn as nn

from .Module.EncoderLayer import EncoderLayer
from .Module.KTEmbedLayer import KTEmbedLayer


class SimpleKT(nn.Module):
    model_name = "SimpleKT"
    use_question = True

    def __init__(self, params, objects):
        super(SimpleKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        num_concept = encoder_config["num_concept"]
        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        dim_final_fc2 = encoder_config["dim_final_fc2"]
        separate_qa = encoder_config["separate_qa"]
        dropout = encoder_config["dropout"]

        self.embed_question_difficulty = self.get_embed_question_diff()
        self.embed_concept_variation = nn.Embedding(num_concept, dim_model)
        self.embed_concept = nn.Embedding(num_concept, dim_model)
        self.embed_interaction_variation = nn.Embedding(2, dim_model)
        if separate_qa:
            # 直接用一个embedding表示在所有concept的interaction
            self.embed_interaction = nn.Embedding(2 * num_concept, dim_model)
        else:
            # 只表示interaction，具体到concept，用concept embedding加interaction embedding表示这个concept的interaction
            self.embed_interaction = nn.Embedding(2, dim_model)

        self.reset()
        self.encoder_layer = EncoderLayer(params, objects)

        self.predict_layer = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, dim_final_fc2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc2, 1),
            nn.Sigmoid()
        )

    def get_embed_question_diff(self):
        difficulty_scalar = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["difficulty_scalar"]
        num_question = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["num_question"]
        dim_model = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["dim_model"]
        if difficulty_scalar:
            # 题目难度用一个标量表示
            embed = nn.Embedding(num_question, 1)
        else:
            # 题目难度用一个embedding表示
            embed = nn.Embedding(num_question, dim_model)

        return embed

    def reset(self):
        num_question = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["num_question"]
        for p in self.parameters():
            # 这一步至关重要，没有执行constant_初始化直接掉点 也就是一开始必须初始化所有习题难度（向量）为0
            if p.size(0) == num_question:
                nn.init.constant_(p, 0.)

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
        separate_qa = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["separate_qa"]
        correct_seq = batch["correct_seq"]
        # c_ct
        concept_emb = self.get_concept_emb(batch)
        if separate_qa:
            # todo: 有问题，如果是only question也要融合interaction_emb
            concept_seq = batch["concept_seq"]
            interaction_seqs = concept_seq + self.num_concept * correct_seq
            interaction_emb = self.embed_interaction(interaction_seqs)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_interaction(correct_seq) + concept_emb

        return concept_emb, interaction_emb

    def forward(self, batch):
        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.get_concept_variation_emb(batch)
        # mu_{q_t}
        question_difficulty_emb = self.embed_question_difficulty(question_seq)
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_interaction_variation(correct_seq)
        # e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}
        interaction_emb = (interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb))

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

        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
