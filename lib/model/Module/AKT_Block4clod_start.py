import torch
import torch.nn as nn
import numpy as np


from .attention import attention_AKT4cold_start, attention_AKT


class Architecture(nn.Module):
    def __init__(self, params):
        super(Architecture, self).__init__()
        self.params = params

        encoder_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"][encoder_type]
        num_block = encoder_config["num_block"]

        # question encoder的偶数层（从0开始）用于对question做self attention，奇数层用于对question和interaction做cross attention
        self.question_encoder = nn.ModuleList([TransformerLayer(params) for _ in range(num_block * 2)])
        self.knowledge_encoder = nn.ModuleList([TransformerLayer(params) for _ in range(num_block)])

    def get_latent(self, batch):
        y = batch["interaction_emb"]
        for block in self.knowledge_encoder:
            y = block(y, y, y, batch["question_difficulty_emb"], apply_pos=False, mask_flag=True)
        return y

    def forward(self, batch):
        x = batch["question_emb"]
        y = batch["interaction_emb"]
        question_difficulty_emb = batch["question_difficulty_emb"]

        for block in self.knowledge_encoder:
            # 对0～t-1时刻前的qa信息进行编码, \hat{y_t}
            y = block(y, y, y, question_difficulty_emb, apply_pos=True, mask_flag=True)

        flag_first = True
        for block in self.question_encoder:
            if flag_first:
                # peek current question
                # False: 没有FFN, 第一层只有self attention, \hat{x_t}
                x = block(x, x, x, question_difficulty_emb, apply_pos=False, mask_flag=True)
                flag_first = False
            else:
                # don't peek current response
                # True: +FFN+残差+layer norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                x = block(x, x, y, question_difficulty_emb, apply_pos=True, mask_flag=False)
                flag_first = True

        return x


class TransformerLayer(nn.Module):
    def __init__(self, params):
        super(TransformerLayer, self).__init__()
        self.params = params

        encoder_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"][encoder_type]
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

    def forward(self, query, key, value, question_difficulty_emb, apply_pos, mask_flag):
        seq_len, batch_size = query.size(1), query.size(0)
        # 上三角和对角为1，其余为0的矩阵
        upper_triangle_ones = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask_flag).astype('uint8')
        src_mask = (torch.from_numpy(upper_triangle_ones) == 0).to(self.params["device"])
        if not mask_flag:
            # 只看过去
            query2 = self.masked_attn_head(query, key, value, src_mask, True, question_difficulty_emb)
        else:
            # 看当前和过去
            query2 = self.masked_attn_head(query, key, value, src_mask, False, question_difficulty_emb)

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
        super(MultiHeadAttention, self).__init__()
        self.params = params

        encoder_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"][encoder_type]
        dim_model = encoder_config["dim_model"]
        key_query_same = encoder_config["key_query_same"]
        num_head = encoder_config["num_head"]
        dropout = encoder_config["dropout"]
        cold_start_step1 = encoder_config["cold_start_step1"]
        cold_start_step2 = encoder_config["cold_start_step2"]
        effect_start_step2 = encoder_config["effect_start_step2"]

        self.dim_model = dim_model
        self.dim_feature = dim_model // num_head
        self.num_head = num_head
        self.key_query_same = key_query_same
        self.cold_start_step1 = cold_start_step1
        self.cold_start_step2 = cold_start_step2
        self.effect_start_step2 = effect_start_step2

        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if not key_query_same:
            self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(num_head, 1, 1))

        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        if self.key_query_same is False:
            nn.init.xavier_uniform_(self.query_linear.weight)

        if self.proj_bias:
            nn.init.constant_(self.key_linear.bias, 0.)
            nn.init.constant_(self.value_linear.bias, 0.)
            if self.key_query_same is False:
                nn.init.constant_(self.query_linear.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, question_difficulty_emb):
        batch_size = q.size(0)
        k = self.key_linear(k).view(batch_size, -1, self.num_head, self.dim_feature)
        if not self.key_query_same:
            q = self.query_linear(q).view(batch_size, -1, self.num_head, self.dim_feature)
        else:
            q = self.key_linear(q).view(batch_size, -1, self.num_head, self.dim_feature)
        v = self.value_linear(v).view(batch_size, -1, self.num_head, self.dim_feature)

        # transpose to get dimensions batch_size * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        q_cold_start1 = q[:, :, :self.cold_start_step1]
        q_cold_start2 = q[:, :, :self.cold_start_step2]
        k_cold_start1 = k[:, :, :self.cold_start_step1]
        k_cold_start2 = k[:, :, :self.cold_start_step2]
        v_cold_start1 = v[:, :, :self.cold_start_step1]
        v_cold_start2 = v[:, :, :self.cold_start_step2]
        upper_triangle_ones_cold_start1 = np.triu(
            np.ones((1, 1, self.cold_start_step1, self.cold_start_step1)), k=not zero_pad
        ).astype('uint8')
        mask_cold_start1 = (torch.from_numpy(upper_triangle_ones_cold_start1) == 0).to(self.params["device"])
        upper_triangle_ones_cold_start2 = np.triu(
            np.ones((1, 1, self.cold_start_step2, self.cold_start_step2)), k=not zero_pad
        ).astype('uint8')
        mask_cold_start2 = (torch.from_numpy(upper_triangle_ones_cold_start2) == 0).to(self.params["device"])
        if question_difficulty_emb is not None:
            q_diff_emb_cold_start1 = question_difficulty_emb[:, :self.cold_start_step1]
            q_diff_emb_cold_start2 = question_difficulty_emb[:, :self.cold_start_step2]
        else:
            q_diff_emb_cold_start1 = None
            q_diff_emb_cold_start2 = None
        scores_cold_start1 = attention_AKT4cold_start(q_cold_start1, k_cold_start1, v_cold_start1,
                                                      self.dim_feature, mask_cold_start1, self.dropout, zero_pad, gammas,
                                                      q_diff_emb_cold_start1, device=self.params["device"],
                                                      cold_start_step1=self.cold_start_step1,
                                                      cold_start_step2=self.cold_start_step2,
                                                      effect_start_step2=self.effect_start_step2)
        scores_cold_start2 = attention_AKT4cold_start(q_cold_start2, k_cold_start2, v_cold_start2,
                                                      self.dim_feature, mask_cold_start2, self.dropout, zero_pad, gammas,
                                                      q_diff_emb_cold_start2, device=self.params["device"],
                                                      cold_start_step1=self.cold_start_step1,
                                                      cold_start_step2=self.cold_start_step2,
                                                      effect_start_step2=self.effect_start_step2)
        scores_warm_start = attention_AKT(q, k, v, self.dim_feature, mask, self.dropout, zero_pad, gammas,
                                          question_difficulty_emb, device=self.params["device"])

        # concatenate heads and put through final linear layer
        concat_cold_start1 = scores_cold_start1.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        output_cold_start1 = self.out_proj(concat_cold_start1)
        concat_cold_start2 = scores_cold_start2.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        output_cold_start2 = self.out_proj(concat_cold_start2)
        concat_warm_start = scores_warm_start.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        output_warm_start = self.out_proj(concat_warm_start)
        output = torch.cat(
            (output_cold_start1,
             output_cold_start2[:, self.cold_start_step1: self.cold_start_step2],
             output_warm_start[:, self.cold_start_step2:]),
            dim=1
        )

        return output
