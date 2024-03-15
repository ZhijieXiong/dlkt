import torch
from torch.nn import Module, Parameter, Linear, GELU, LayerNorm, Dropout
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import numpy as np

from .attention import attention_CL4KT


class CL4KTTransformerLayer(Module):
    def __init__(self, params, objects):
        super(CL4KTTransformerLayer, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
        dim_model = encoder_config["dim_model"]
        dim_ff = encoder_config["dim_ff"]
        dropout = encoder_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttentionWithIndividualFeatures(params, objects)

        # Two layer norm and two dropout layers
        self.dropout1 = Dropout(dropout)
        self.layer_norm1 = LayerNorm(dim_model)

        self.linear1 = Linear(dim_model, dim_ff)
        self.activation = GELU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_ff, dim_model)
        self.dropout2 = Dropout(dropout)
        self.layer_norm2 = LayerNorm(dim_model)

    def forward(self, mask, query, key, values, apply_pos=True):
        # mask: 0 means that it can peek (留意) only past values. 1 means that block can peek current and past values

        batch_size, seq_len = query.size(0), query.size(1)
        # 从输入矩阵中抽取上三角矩阵，k表示从第几条对角线开始
        upper_tri_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(upper_tri_mask) == 0).to(self.params["device"])
        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            # 单向的attention，只看过去
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        elif mask == 1:
            # 单向的attention，包括当前
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        else:
            # 双向的attention
            query2, attn = self.masked_attn_head(query, key, values, mask=bert_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query, attn


class MultiHeadAttentionWithIndividualFeatures(Module):
    def __init__(self, params, objects):
        super(MultiHeadAttentionWithIndividualFeatures, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
        dim_model = encoder_config["dim_model"]
        num_head = encoder_config["num_head"]
        key_query_same = encoder_config["key_query_same"]
        dropout = encoder_config["dropout"]

        self.bias = True
        self.proj_bias = self.bias
        self.v_linear = Linear(dim_model, dim_model, bias=self.bias)
        self.k_linear = Linear(dim_model, dim_model, bias=self.bias)
        if not key_query_same:
            self.q_linear = Linear(dim_model, dim_model, bias=self.bias)
        self.dropout = Dropout(dropout)
        self.out_proj = Linear(dim_model, dim_model, bias=self.bias)
        self.gammas = Parameter(torch.zeros(num_head, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
        key_query_same = encoder_config["key_query_same"]

        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not key_query_same:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if not key_query_same:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad=True):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["CL4KT"]
        dim_model = encoder_config["dim_model"]
        num_head = encoder_config["num_head"]
        key_query_same = encoder_config["key_query_same"]
        dim_head = dim_model // num_head

        batch_size = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(batch_size, -1, num_head, dim_head)
        if key_query_same is False:
            q = self.q_linear(q).view(batch_size, -1, num_head, dim_head)
        else:
            q = self.k_linear(q).view(batch_size, -1, num_head, dim_head)
        v = self.v_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions batch_size * num_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, attn_scores = attention_CL4KT(
            q, k, v, dim_head, mask, self.dropout, self.params["device"], self.gammas, zero_pad
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
        output = self.out_proj(concat)

        return output, attn_scores
