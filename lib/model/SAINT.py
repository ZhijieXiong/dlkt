import torch
import torch.nn as nn

from torch.nn import Dropout

from .util import transformer_FFN, get_clones, ut_mask, encode_position


class SAINT(nn.Module):
    model_name = "SAINT"

    def __init__(self, params, objects):
        super(SAINT, self).__init__()
        self.params = params
        self.objects = objects

        backbone_config = self.params["models_config"]["kt_model"]["backbone"]["SAINT"]
        seq_len = backbone_config["seq_len"]
        dim_emb = backbone_config["dim_emb"]
        dropout = backbone_config["dropout"]
        num_block = backbone_config["num_block"]

        self.embed_position = nn.Embedding(seq_len, dim_emb)
        self.encoder = get_clones(Encoder_block(params, objects), num_block)
        self.decoder = get_clones(Decoder_block(params, objects), num_block)
        self.dropout = Dropout(dropout)
        self.out = nn.Linear(dim_emb, 1)

    def forward(self, batch):
        backbone_config = self.params["models_config"]["kt_model"]["backbone"]["SAINT"]
        num_block = backbone_config["num_block"]
        question_seq = batch["question_seq"]
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"][:, :-1]

        position_seq = encode_position(question_seq.shape[1], self.params["device"])
        position_emb = self.embed_position(position_seq)

        # pass through each of the encoder blocks in sequence
        question_emb = question_seq
        for i in range(num_block):
            question_emb = self.encoder[i](question_emb, concept_seq, position_emb, first_block=i == 0)
            concept_seq = question_emb

        # pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(correct_seq.shape[0], 1).to(self.params["device"])
        correct_seq = torch.cat((start_token, correct_seq), dim=-1)

        correct_emb = correct_seq
        for i in range(num_block):
            correct_emb = self.decoder[i](correct_emb, position_emb, encoder_out=question_emb, first_block=i == 0)

        res = self.out(self.dropout(correct_emb))
        res = torch.sigmoid(res).squeeze(-1)

        return res

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

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


class Encoder_block(nn.Module):
    def __init__(self, params, objects):
        super(Encoder_block, self).__init__()
        self.params = params
        self.objects = objects

        backbone_config = self.params["models_config"]["kt_model"]["backbone"]["SAINT"]
        num_question = backbone_config["num_question"]
        num_concept = backbone_config["num_concept"]
        dim_emb = backbone_config["dim_emb"]
        num_attn_head = backbone_config["num_attn_head"]
        dropout = backbone_config["dropout"]

        # embedding  q,k,v = E = exercise ID embedding + category embedding + and position embedding.
        self.embed_question = nn.Embedding(num_question, embedding_dim=dim_emb)
        self.embed_concept = nn.Embedding(num_concept, embedding_dim=dim_emb)

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=num_attn_head, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_emb)
        self.dropout1 = Dropout(dropout)
        self.ffn_en = transformer_FFN(dim_emb, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_emb)
        self.dropout2 = Dropout(dropout)

    def forward(self, question_seq, concept_seq, position_emb, first_block=True):
        if first_block:
            out = self.embed_concept(concept_seq) + self.embed_question(question_seq) + position_emb
        else:
            out = question_seq
        out = out.permute(1, 0, 2)

        # norm -> attn -> drop -> skip corresponding to transformers' norm_first
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        # attention mask upper triangular
        out, attn_wt = self.multi_en(out, out, out, attn_mask=ut_mask(n, self.params["device"]))
        out = self.dropout1(out)
        # skip connection
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out

        return out


class Decoder_block(nn.Module):
    def __init__(self, params, objects):
        super(Decoder_block, self).__init__()
        self.params = params
        self.objects = objects

        backbone_config = self.params["models_config"]["kt_model"]["backbone"]["SAINT"]
        dim_emb = backbone_config["dim_emb"]
        num_attn_head = backbone_config["num_attn_head"]
        dropout = backbone_config["dropout"]

        # response embedding, include a start token
        self.embed_correct = nn.Embedding(2 + 1, embedding_dim=dim_emb)
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=num_attn_head, dropout=dropout)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=num_attn_head, dropout=dropout)
        self.ffn_en = transformer_FFN(dim_emb, dropout)
        self.layer_norm1 = nn.LayerNorm(dim_emb)
        self.layer_norm2 = nn.LayerNorm(dim_emb)
        self.layer_norm3 = nn.LayerNorm(dim_emb)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, correct_seq, position_emb, encoder_out, first_block=True):
        if first_block:
            correct_emb = self.embed_correct(correct_seq)
            out = correct_emb + position_emb
        else:
            out = correct_seq

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, attn_mask=ut_mask(n, self.params["device"]))
        out = self.dropout1(out)
        out = skip_out + out

        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.layer_norm2(encoder_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, encoder_out, encoder_out, attn_mask=ut_mask(n, self.params["device"]))
        out = self.dropout2(out)
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout3(out)
        out = out + skip_out

        return out
