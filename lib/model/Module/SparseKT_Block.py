import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import numpy as np

from .attention import attention_SparseKT


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(1), :]  # ( 1,seq,  Feature)


class Architecture(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        dim_model = encoder_config["dim_model"]
        num_block = encoder_config["num_block"]
        seq_len = encoder_config["seq_len"]

        self.blocks_2 = nn.ModuleList([TransformerLayer(params) for _ in range(num_block)])
        self.position_emb = CosinePositionalEmbedding(d_model=dim_model, max_len=seq_len)

    def forward(
        self,
        question_emb,
        interaction_emb,
    ):
        question_po_emb = self.position_emb(question_emb)
        question_emb = question_emb + question_po_emb
        interaction_pos_emb = self.position_emb(interaction_emb)
        interaction_emb = interaction_emb + interaction_pos_emb

        qa_pos_embed = interaction_emb
        q_pos_embed = question_emb

        y = qa_pos_embed
        x = q_pos_embed

        for block in self.blocks_2:
            # True: +FFN+残差+layer norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
            x, attn_weights = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                apply_pos=True,
            )
        return x


class TransformerLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
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

    def forward(
        self,
        mask,
        query,
        key,
        values,
        apply_pos=True,
    ):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """
        seq_len, batch_size = query.size(1), query.size(0)
        no_peek_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(no_peek_mask) == 0).to(self.params["device"])
        if mask == 0:  # If 0, zero-padding is needed.
            # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
            query2, _ = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2, _ = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(
                self.dropout(self.activation(self.linear1(query)))  # FFN
            )
            query = query + self.dropout2(query2)  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query, _


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        dim_model = encoder_config["dim_model"]
        kq_same = encoder_config["kq_same"]
        num_head = encoder_config["num_head"]
        dropout = encoder_config["dropout"]
        dim_feature = dim_model // num_head
        bias = True

        self.d_model = dim_model
        self.dim_feature = dim_feature
        self.num_head = num_head
        self.kq_same = kq_same

        self.v_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.k_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        kq_same = encoder_config["kq_same"]

        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["SparseKT"]
        dim_model = encoder_config["dim_model"]
        kq_same = encoder_config["kq_same"]
        num_head = encoder_config["num_head"]
        k_index = encoder_config["k_index"]
        dim_feature = dim_model // num_head

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, num_head, dim_feature)
        if kq_same is False:
            q = self.q_linear(q).view(bs, -1, num_head, dim_feature)
        else:
            q = self.k_linear(q).view(bs, -1, num_head, dim_feature)
        v = self.v_linear(v).view(bs, -1, num_head, dim_feature)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, attn_weights = attention_SparseKT(
            q,
            k,
            v,
            dim_feature,
            mask,
            self.dropout,
            zero_pad,
            k_index,
            self.params["device"]
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, attn_weights
