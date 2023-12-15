import torch
import torch.nn as nn
import numpy as np

from .util import CosinePositionalEmbedding
from .attention import attention_SimpleKT


class Architecture(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        num_block = encoder_config["num_block"]
        dim_model = encoder_config["dim_model"]
        seq_len = encoder_config["seq_len"]

        self.dim_model = dim_model
        self.blocks = nn.ModuleList([TransformerLayer(params) for _ in range(num_block)])
        self.position_emb = CosinePositionalEmbedding(d_model=self.dim_model, max_len=seq_len)

    def forward(self, batch):
        question_emb = batch["question_emb"]
        interaction_emb = batch["interaction_emb"]

        # target shape: (batch_size, seq_len)
        emb_position_concept = self.position_emb(question_emb)
        question_emb = question_emb + emb_position_concept
        emb_position_interaction = self.position_emb(interaction_emb)
        interaction_emb = interaction_emb + emb_position_interaction

        y = interaction_emb
        x = question_emb

        for block in self.blocks:
            # apply_pos is True: FFN+残差+lay norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，第一题只有question信息，无interaction信息
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        dim_model = encoder_config["dim_model"]
        dim_ff = encoder_config["dim_ff"]
        dropout = encoder_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(params)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        batch_size, seq_len = query.size(0), query.size(1)
        # 上三角和对角为1，其余为0的矩阵
        upper_triangle_ones = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype('uint8')
        # 用于取矩阵下三角
        src_mask = (torch.from_numpy(upper_triangle_ones) == 0).to(self.params["device"])
        if mask == 0:
            # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        # 残差
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, params, bias=True):
        super().__init__()

        self.params = params
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        dim_model = encoder_config["dim_model"]
        dropout = encoder_config["dropout"]
        key_query_same = encoder_config["key_query_same"]

        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if not key_query_same:
            self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.bias_projection = bias
        self.projection_out = nn.Linear(dim_model, dim_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        key_query_same = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]["key_query_same"]
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        if not key_query_same:
            nn.init.xavier_uniform_(self.query_linear.weight)

        if self.bias_projection:
            nn.init.constant_(self.key_linear.bias, 0.)
            nn.init.constant_(self.value_linear.bias, 0.)
            if key_query_same is False:
                nn.init.constant_(self.query_linear.bias, 0.)
            nn.init.constant_(self.projection_out.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SimpleKT"]
        key_query_same = encoder_config["key_query_same"]
        num_head = encoder_config["num_head"]
        dim_model = encoder_config["dim_model"]
        dim_head = dim_model // num_head
        batch_size = q.size(0)

        k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
        if key_query_same:
            q = self.key_linear(q).view(batch_size, -1, num_head, dim_head)
        else:
            q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
        v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions (batch_size * num_head * seq_len * dim_model)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention_SimpleKT(q, k, v, dim_head, mask, self.dropout, zero_pad, device=self.params["device"])

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
        output = self.projection_out(concat)

        return output
