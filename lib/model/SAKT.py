import torch
import torch.nn as nn
from copy import deepcopy

from .Module.KTEmbedLayer import KTEmbedLayer


def position_encode(seq_len, device):
    return torch.arange(seq_len).unsqueeze(0).to(device)


def ut_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class transformer_FFN(nn.Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_size, self.emb_size),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


class SAKT(nn.Module):
    model_name = "SAKT"

    def __init__(self, params, objects):
        super(SAKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SAKT"]
        num_concept = encoder_config["num_concept"]
        dim_emb = encoder_config["dim_emb"]
        dropout = encoder_config["dropout"]
        num_block = encoder_config["num_block"]
        seq_len = encoder_config["seq_len"]

        self.embed_interaction = nn.Embedding(num_concept * 2, dim_emb)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_position = nn.Embedding(seq_len, dim_emb)

        self.blocks = get_clones(Blocks(params), num_block)

        self.dropout_layer = nn.Dropout(dropout)
        self.out = nn.Linear(dim_emb, 1)

    def forward(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SAKT"]
        num_concept = encoder_config["num_concept"]
        num_block = encoder_config["num_block"]
        correct_seq = batch["correct_seq"]

        if data_type == "single_concept":
            # 历史的interaction
            concept_seq = batch["concept_seq"]
            interaction_seq = concept_seq[:, :-1] + num_concept * correct_seq[:, :-1]
            interaction_emb = self.embed_interaction(interaction_seq)
            # 未来的concept
            concept_emb = self.embed_concept(concept_seq[:, 1:])
        else:
            question_seq = batch["question_seq"]
            interaction_emb = KTEmbedLayer.interaction_fused_emb(
                self.embed_interaction,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                question_seq,
                correct_seq,
                num_concept,
                fusion_type="mean"
            )[:, :-1]
            concept_emb = KTEmbedLayer.concept_fused_emb(
                self.embed_concept,
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                question_seq,
                fusion_type="mean"
            )[:, 1:]

        position_emb = self.embed_position(position_encode(interaction_emb.shape[1], self.params["device"]))
        interaction_emb = interaction_emb + position_emb

        for i in range(num_block):
            interaction_emb = self.blocks[i](concept_emb, interaction_emb, interaction_emb)

        p = torch.sigmoid(self.out(self.dropout_layer(interaction_emb))).squeeze(-1)

        return p

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

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


class Blocks(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SAKT"]
        dim_emb = encoder_config["dim_emb"]
        num_head = encoder_config["num_head"]
        dropout = encoder_config["dropout"]

        self.attn = nn.MultiheadAttention(dim_emb, num_head, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(dim_emb)

        self.FFN = transformer_FFN(dim_emb, dropout)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_layer_norm = nn.LayerNorm(dim_emb)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(k.shape[0], self.params["device"])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)

        return emb
