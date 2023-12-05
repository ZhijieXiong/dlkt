from .BaseModel4CL import BaseModel4CL
from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .loss_util import *
from .util import *


class Block(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["qSAKT"]
        dim_emb = encoder_config["dim_emb"]
        num_head = encoder_config["num_head"]
        dropout = encoder_config["dropout"]

        self.attn = nn.MultiheadAttention(dim_emb, num_head, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(dim_emb)

        self.FFN = nn.Sequential(
            nn.Linear(dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb),
            nn.Dropout(dropout)
        )
        self.FFN_layer_norm = nn.LayerNorm(dim_emb)

    def forward(self, q, k, v):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        seq_len = k.shape[0]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.params["device"])

        # transformer default: attn -> drop -> skip -> norm
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_layer_norm(attn_emb + emb)

        return emb


class SAKT(nn.Module):
    model_name = "SAKT"

    def __init__(self, params, objects):
        super(SAKT, self).__init__()

        pass

    def get_concept_emb(self):
        return self.embed_concept.weight

    def forward(self, batch):
        pass
