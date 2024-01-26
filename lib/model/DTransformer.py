import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from .Module.attention import attention_DTransformer


class DTransformer(nn.Module):
    model_name = "DTransformer"
    MIN_SEQ_LEN = 5

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        num_concept = encoder_config["num_concept"]
        num_question = encoder_config["num_question"]
        dim_model = encoder_config["dim_model"]
        dim_final_fc = encoder_config["dim_final_fc"]
        num_head = encoder_config["num_head"]
        num_concept_prototype = encoder_config["num_concept_prototype"]
        dropout = encoder_config["dropout"]
        proj = encoder_config["proj"]

        self.embed_concept = nn.Embedding(num_concept, dim_model)
        self.embed_correct = nn.Embedding(2, dim_model)

        if num_question > 0:
            self.embed_concept_variation = nn.Embedding(num_concept, dim_model)
            self.embed_correct_variation = nn.Embedding(2, dim_model)
            self.embed_question_difficulty = nn.Embedding(num_question, 1)

        self.num_head = num_head
        # 前三个都是AKT中的
        # 提取习题表征（习题embedding作为k、q、v）
        self.question_encoder = DTransformerLayer(params)
        # 提取交互表征（交互，即习题和回答结果，的embedding作为k、q、v）
        self.knowledge_encoder = DTransformerLayer(params)
        # 提取知识状态（习题表征作为k和q，交互表征作为v）
        self.knowledge_retriever = DTransformerLayer(params)
        params_ = deepcopy(params)
        params_["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]["key_query_same"] = False
        self.block4 = DTransformerLayer(params_)

        self.num_concept_prototype = num_concept_prototype
        self.knowledge_params = nn.Parameter(torch.empty(num_concept_prototype, dim_model))
        torch.nn.init.uniform_(self.knowledge_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, dim_final_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc // 2, 1),
        )

        if proj:
            self.proj = nn.Sequential(nn.Linear(dim_model, dim_model), nn.GELU())
        else:
            self.proj = None

    def forward(self, emb_concept, emb_interaction, seqs_length):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        num_layer = encoder_config["num_layer"]
        shortcut = encoder_config["shortcut"]

        if shortcut:
            # AKT, no read-out module
            question_representation, _ = self.question_encoder(emb_concept, emb_concept, emb_concept, seqs_length,
                                                               peek_cur=True)
            interaction_representation, scores = self.knowledge_encoder(emb_interaction, emb_interaction,
                                                                        emb_interaction, seqs_length, peek_cur=True)
            return self.knowledge_retriever(question_representation, question_representation,
                                            interaction_representation, seqs_length, peek_cur=False), scores, None

        if num_layer == 1:
            question_representation = emb_concept
            knowledge_representation, q_scores = self.question_encoder(emb_concept, emb_concept, emb_interaction, seqs_length,
                                                                       peek_cur=True)
        elif num_layer == 2:
            question_representation = emb_concept
            interaction_representation, _ = self.question_encoder(emb_interaction, emb_interaction, emb_interaction,
                                                                  seqs_length, peek_cur=True)
            knowledge_representation, q_scores = self.knowledge_encoder(question_representation,
                                                                        question_representation,
                                                                        interaction_representation,
                                                                        seqs_length, peek_cur=True)
        else:
            # 融合了习题难度的concept embedding作为k、q、v提取有上下文信息的question embedding（论文中的m）
            question_representation, _ = self.question_encoder(emb_concept, emb_concept, emb_concept, seqs_length,
                                                               peek_cur=True)
            # interaction embedding作为k、q、v提取有上下文信息的interaction embedding
            interaction_representation, _ = self.knowledge_encoder(emb_interaction, emb_interaction, emb_interaction,
                                                                   seqs_length, peek_cur=True)
            # question embedding作为k、q，interaction embedding作为v，提取学生的知识状态（AKT），该状态经过read-out得到论文的z
            knowledge_representation, q_scores = self.knowledge_retriever(question_representation,
                                                                          question_representation,
                                                                          interaction_representation,
                                                                          seqs_length, peek_cur=True)

        batch_size, seq_len, dim_model = knowledge_representation.size()
        dim_knowledge = self.num_concept_prototype

        # query是论文中的K，作为attention的q。该模型用一个(dim_knowledge, dim_model)的tensor来表征底层的知识状态
        K = (
            self.knowledge_params[None, :, None, :]
            .expand(batch_size, -1, seq_len, -1)
            .contiguous()
            .view(batch_size * dim_knowledge, seq_len, dim_model)
        )
        question_representation = question_representation.unsqueeze(1).expand(-1, dim_knowledge, -1, -1).reshape_as(K)
        knowledge_representation = knowledge_representation.unsqueeze(1).expand(-1, dim_knowledge, -1, -1).reshape_as(K)

        # 论文中的z和attention的score（有MaxOut的attention）
        z, k_scores = self.block4(
            K, question_representation, knowledge_representation, torch.repeat_interleave(seqs_length, dim_knowledge),
            peek_cur=False
        )
        z = (
            z.view(batch_size, dim_knowledge, seq_len, dim_model)  # unpack dimensions
            .transpose(1, 2)  # (batch_size, seq_len, dim_knowledge, dim_model)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        k_scores = (
            k_scores.view(batch_size, dim_knowledge, self.num_head, seq_len, seq_len)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (batch_size, n_heads, seq_len, dim_knowledge, seq_len)
            .contiguous()
        )
        return z, q_scores, k_scores

    def embedding(self, concept_seq, correct_seq, question_seq=None):
        seqs_length = (correct_seq >= 0).sum(dim=1)
        # set prediction mask
        concept_seq = concept_seq.masked_fill(concept_seq < 0, 0)
        correct_seq = correct_seq.masked_fill(correct_seq < 0, 0)

        emb_concept = self.embed_concept(concept_seq)
        emb_interaction = self.embed_correct(correct_seq) + emb_concept

        question_difficulty = 0.0

        if question_seq is not None:
            question_seq = question_seq.masked_fill(question_seq < 0, 0)
            question_difficulty = self.embed_question_difficulty(question_seq)

            emb_concept_variation = self.embed_concept_variation(concept_seq)
            # AKT question encoder 输入
            emb_concept = emb_concept + emb_concept_variation * question_difficulty

            emb_correct_variation = self.embed_correct_variation(correct_seq) + emb_concept_variation
            # AKT knowledge encoder 输入
            emb_interaction = emb_interaction + emb_correct_variation * question_difficulty

        return emb_concept, emb_interaction, seqs_length, question_difficulty

    def readout(self, z, emb_concept):
        # 使用当前时刻的concept作为q，学习到的K作为k，提取的知识状态z作为v，做attention，得到融合后的知识状态
        batch_size, seq_len, _ = emb_concept.size()
        key = (
            self.knowledge_params[None, None, :, :]
            .expand(batch_size, seq_len, -1, -1)
            .view(batch_size * seq_len, self.num_concept_prototype, -1)
        )
        value = z.reshape(batch_size * seq_len, self.num_concept_prototype, -1)

        beta = torch.matmul(
            key,
            emb_concept.reshape(batch_size * seq_len, -1, 1),
        ).view(batch_size * seq_len, 1, self.num_concept_prototype)
        alpha = torch.softmax(beta, -1)
        # 论文公式(19)
        return torch.matmul(alpha, value).view(batch_size, seq_len, -1)

    def predict(self, concept_seq, correct_seq, question_seq=None, n=1):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        shortcut = encoder_config["shortcut"]

        # emb_concept是融合了习题难度的concept，所以实际上可以看做question的embedding
        emb_concept, emb_interaction, seqs_length, question_difficulty = \
            self.embedding(concept_seq, correct_seq, question_seq)
        z, q_scores, k_scores = self(emb_concept, emb_interaction, seqs_length)

        query = emb_concept[:, n - 1:, :]
        if shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            knowledge_state = z
        else:
            # predict T+N，即论文中公式(19)的z_{q_t}
            knowledge_state = self.readout(z[:, : query.size(1), :], query)

        y = self.out(torch.cat([query, knowledge_state], dim=-1)).squeeze(-1)

        if question_seq is not None:
            return y, z, emb_concept, (question_difficulty ** 2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return y, z, emb_concept, 0.0, (q_scores, k_scores)

    def get_loss(self, concept_seq, correct_seq, question_seq=None):
        # reg_loss实际上就是question difficulty embedding的二范数，和AKT一样，作为loss的一部分
        logits, _, _, reg_loss, _ = self.predict(concept_seq, correct_seq, question_seq)
        masked_labels = correct_seq[correct_seq >= 0].float()
        masked_logits = logits[correct_seq >= 0]
        return (
                F.binary_cross_entropy_with_logits(
                    masked_logits, masked_labels, reduction="mean"
                )
                + reg_loss
        )

    def get_cl_loss(self, concept_seq, correct_seq, question_seq=None):
        use_hard_neg = self.params["other"]["DTransformer"]["use_hard_neg"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dropout = encoder_config["dropout"]
        window = encoder_config["window"]

        batch_size = correct_seq.size(0)

        # skip CL for batches that are too short
        seqs_length = (correct_seq >= 0).sum(dim=1)
        min_len = seqs_length.min().item()
        if min_len < DTransformer.MIN_SEQ_LEN:
            return self.get_loss(concept_seq, correct_seq, question_seq)

        # augmentation
        concept_seq_ = concept_seq.clone()
        correct_seq_ = correct_seq.clone()

        if question_seq is not None:
            question_seq_ = question_seq.clone()
        else:
            question_seq_ = None

        # manipulate order
        for b in range(batch_size):
            idx = random.sample(
                range(seqs_length[b] - 1), max(1, int(seqs_length[b] * dropout))
            )
            for i in idx:
                concept_seq_[b, i], concept_seq_[b, i + 1] = concept_seq_[b, i + 1], concept_seq_[b, i]
                correct_seq_[b, i], correct_seq_[b, i + 1] = correct_seq_[b, i + 1], correct_seq_[b, i]
                if question_seq_ is not None:
                    question_seq_[b, i], question_seq_[b, i + 1] = question_seq_[b, i + 1], question_seq_[b, i]

        # hard negative
        s_flip = correct_seq.clone() if use_hard_neg else correct_seq_
        for b in range(batch_size):
            # manipulate score
            idx = random.sample(
                range(seqs_length[b]), max(1, int(seqs_length[b] * dropout))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not use_hard_neg:
            correct_seq_ = s_flip

        # z就是论文中的z_{q_t}
        logits, z_1, q_emb, reg_loss, _ = self.predict(concept_seq, correct_seq, question_seq)
        masked_logits = logits[correct_seq >= 0]

        # positive
        _, z_2, *_ = self.predict(concept_seq_, correct_seq_, question_seq_)
        input_ = self.sim(z_1[:, :min_len, :], z_2[:, :min_len, :])

        if use_hard_neg:
            # negative
            _, z_3, *_ = self.predict(concept_seq, s_flip, question_seq)
            hard_neg = self.sim(z_1[:, :min_len, :], z_3[:, :min_len, :])
            input_ = torch.cat((input_, hard_neg), dim=1)

        target = (
            torch.arange(correct_seq.size(0))[:, None]
            .to(self.knowledge_params.device)
            .expand(-1, min_len)
        )
        cl_loss = F.cross_entropy(input_, target)

        # prediction loss
        masked_labels = correct_seq[correct_seq >= 0].float()
        prediction_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        for i in range(1, window):
            label = correct_seq[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            prediction_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        prediction_loss /= window

        weight_cl_loss = self.params["loss_config"]["cl loss"]
        return prediction_loss + cl_loss * weight_cl_loss + reg_loss, prediction_loss, cl_loss

    def sim(self, z1, z2):
        bs, seq_len, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seq_len, self.num_concept_prototype, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seq_len, self.num_concept_prototype, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

    def tracing(self, q, s, pid=None):
        # add fake q, s, pid to generate the last tracing result
        pad = torch.tensor([0]).to(self.knowledge_params.device)
        q = torch.cat([q, pad], dim=0).unsqueeze(0)
        s = torch.cat([s, pad], dim=0).unsqueeze(0)
        if pid is not None:
            pid = torch.cat([pid, pad], dim=0).unsqueeze(0)

        with torch.no_grad():
            # q_emb: (bs, seq_len, d_model)
            # z: (bs, seq_len, n_know * d_model)
            # know_params: (n_know, d_model)->(n_know, 1, d_model)
            q_emb, s_emb, seqs_length, _ = self.embedding(q, s, pid)
            z, _, _ = self(q_emb, s_emb, seqs_length)
            query = self.knowledge_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
            z = z.expand(self.num_concept_prototype, -1, -1).contiguous()
            h = self.readout(z, query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            y = torch.sigmoid(y)

        return y


class DTransformerLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dim_model = encoder_config["dim_model"]
        dropout = encoder_config["dropout"]

        self.masked_attn_head = MultiHeadAttention(params)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, query, key, values, seqs_length, peek_cur=False):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dropout = encoder_config["dropout"]

        # construct mask
        seq_len = query.size(1)
        mask = torch.ones(seq_len, seq_len).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.params["device"])

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                # sample for each batch
                if seqs_length[b] < DTransformer.MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(seqs_length[b] - 1), max(1, int(seqs_length[b] * dropout))
                )
                for i in idx:
                    mask[b, :, i + 1:, i] = 0

        # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, max_out=not peek_cur
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dim_model = encoder_config["dim_model"]
        num_head = encoder_config["num_head"]
        key_query_same = encoder_config["key_query_same"]
        bias = encoder_config["bias"]

        self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if key_query_same:
            self.key_linear = self.query_linear
        else:
            self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(num_head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, max_out=False):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dim_model = encoder_config["dim_model"]
        num_head = encoder_config["num_head"]
        dim_head = dim_model // num_head

        batch_size = q.size(0)
        # perform linear operation and split into num_head
        q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
        k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
        v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions batch_size * num_head * seq_len * dim_head
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        v_, scores = attention_DTransformer(
            q,
            k,
            v,
            mask,
            self.gammas,
            max_out,
        )

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)

        output = self.out_proj(concat)

        return output, scores
