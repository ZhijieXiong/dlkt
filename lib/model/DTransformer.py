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
        num_knowledge_prototype = encoder_config["num_knowledge_prototype"]
        dropout = encoder_config["dropout"]
        proj = encoder_config["proj"]
        use_question = encoder_config["use_question"]

        self.embed_concept = nn.Embedding(num_concept, dim_model)
        self.embed_correct = nn.Embedding(2, dim_model)

        if use_question:
            self.embed_concept_variation = nn.Embedding(num_concept, dim_model)
            self.embed_correct_variation = nn.Embedding(2, dim_model)
            self.embed_question_difficulty = nn.Embedding(num_question, 1)

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

        self.num_knowledge_prototype = num_knowledge_prototype
        self.knowledge_params = nn.Parameter(torch.empty(num_knowledge_prototype, dim_model))
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

    def forward(self, concept_emb, interaction_emb, seqs_length):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        num_layer = encoder_config["num_layer"]
        num_head = encoder_config["num_head"]

        if num_layer == 1:
            question_representation = concept_emb
            knowledge_representation, q_scores = self.question_encoder(concept_emb, concept_emb, interaction_emb, seqs_length,
                                                                       peek_cur=True)
        elif num_layer == 2:
            question_representation = concept_emb
            interaction_representation, _ = self.question_encoder(interaction_emb, interaction_emb, interaction_emb,
                                                                  seqs_length, peek_cur=True)
            knowledge_representation, q_scores = self.knowledge_encoder(question_representation,
                                                                        question_representation,
                                                                        interaction_representation,
                                                                        seqs_length, peek_cur=True)
        else:
            # 融合了习题难度的concept embedding作为k、q、v提取有上下文信息的question embedding（论文中的m）
            question_representation, _ = self.question_encoder(concept_emb, concept_emb, concept_emb, seqs_length,
                                                               peek_cur=True)
            # interaction embedding作为k、q、v提取有上下文信息的interaction embedding
            interaction_representation, _ = self.knowledge_encoder(interaction_emb, interaction_emb, interaction_emb,
                                                                   seqs_length, peek_cur=True)
            # question embedding作为k、q，interaction embedding作为v，提取学生的知识状态（AKT），该状态经过read-out得到论文的z
            knowledge_representation, q_scores = self.knowledge_retriever(question_representation,
                                                                          question_representation,
                                                                          interaction_representation,
                                                                          seqs_length, peek_cur=True)

        batch_size, seq_len, dim_model = knowledge_representation.size()
        num_prototype = self.num_knowledge_prototype

        # query是论文中的K，作为attention的q。该模型用一个(num_prototype, dim_model)的tensor来表征底层的知识状态
        K = (
            self.knowledge_params[None, :, None, :]
            .expand(batch_size, -1, seq_len, -1)
            .contiguous()
            .view(batch_size * num_prototype, seq_len, dim_model)
        )
        question_representation = question_representation.unsqueeze(1).expand(-1, num_prototype, -1, -1).reshape_as(K)
        knowledge_representation = knowledge_representation.unsqueeze(1).expand(-1, num_prototype, -1, -1).reshape_as(K)

        # 论文中的z和attention的score（有MaxOut的attention）
        z, k_scores = self.block4(
            K, question_representation, knowledge_representation, torch.repeat_interleave(seqs_length, num_prototype),
            peek_cur=False
        )
        z = (
            z.view(batch_size, num_prototype, seq_len, dim_model)  # unpack dimensions
            .transpose(1, 2)  # (batch_size, seq_len, num_prototype, dim_model)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        k_scores = (
            k_scores.view(batch_size, num_prototype, num_head, seq_len, seq_len)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (batch_size, n_heads, seq_len, num_prototype, seq_len)
            .contiguous()
        )
        return z, q_scores, k_scores

    def get_latent(self, batch):
        concept_emb, _, _ = self.embed_input(batch)
        z = self.get_z(batch)
        query = concept_emb[:, :, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        return latent

    def get_predict_score(self, batch):
        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]

        # concept_emb是融合了习题难度的concept，所以实际上可以看做question的embedding
        seqs_length = (correct_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, question_difficulty = self.embed_input({
            "concept_seq": concept_seq, "correct_seq": correct_seq, "question_seq": question_seq
        })
        z, q_scores, k_scores = self(concept_emb, interaction_emb, seqs_length)

        n = 1
        query = concept_emb[:, n - 1:, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        predict_score = torch.sigmoid(self.out(torch.cat([query, latent], dim=-1)).squeeze(-1))

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record):
        predict_loss = self.get_predict_loss_(batch, loss_record)
        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        correct_seq = batch["correct_seq"]
        seqs_length = (correct_seq >= 0).sum(dim=1)
        min_len = seqs_length.min().item()
        if min_len < DTransformer.MIN_SEQ_LEN:
            # skip CL for batches that are too short
            loss_record.add_loss("cl loss", 0, 1)
            loss = predict_loss
        else:
            cl_loss = self.get_cl_loss(batch)
            loss_record.add_loss("cl loss", cl_loss.detach().cpu().item(), 1)
            weight_cl_loss = self.params["loss_config"]["cl loss"]
            loss = predict_loss + cl_loss * weight_cl_loss

        return loss

    def get_z(self, batch):
        correct_seq = batch["correct_seq"]
        seqs_length = (correct_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, question_difficulty = self.embed_input(batch)
        z, _, _ = self.forward(concept_emb, interaction_emb, seqs_length)

        return z

    def embed_input(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        use_question = encoder_config["use_question"]

        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]

        # set prediction mask
        concept_seq = concept_seq.masked_fill(concept_seq < 0, 0)
        correct_seq = correct_seq.masked_fill(correct_seq < 0, 0)

        concept_emb = self.embed_concept(concept_seq)
        interaction_emb = self.embed_correct(correct_seq) + concept_emb

        question_diff_emb = 0.0
        if use_question:
            question_seq = question_seq.masked_fill(question_seq < 0, 0)
            question_diff_emb = self.embed_question_difficulty(question_seq)
            concept_variation_emb = self.embed_concept_variation(concept_seq)
            # AKT question encoder 输入
            concept_emb = concept_emb + concept_variation_emb * question_diff_emb
            correct_variation_emb = self.embed_correct_variation(correct_seq) + concept_variation_emb
            # AKT knowledge encoder 输入
            interaction_emb = interaction_emb + correct_variation_emb * question_diff_emb

        return concept_emb, interaction_emb, question_diff_emb

    def readout(self, z, concept_emb):
        # 使用当前时刻的concept作为q，学习到的K作为k，提取的知识状态z作为v，做attention，得到融合后的知识状态
        batch_size, seq_len, _ = concept_emb.size()
        key = (
            self.knowledge_params[None, None, :, :]
            .expand(batch_size, seq_len, -1, -1)
            .view(batch_size * seq_len, self.num_knowledge_prototype, -1)
        )
        value = z.reshape(batch_size * seq_len, self.num_knowledge_prototype, -1)

        beta = torch.matmul(
            key,
            concept_emb.reshape(batch_size * seq_len, -1, 1),
        ).view(batch_size * seq_len, 1, self.num_knowledge_prototype)
        alpha = torch.softmax(beta, -1)
        # 论文公式(19)
        return torch.matmul(alpha, value).view(batch_size, seq_len, -1)

    def predict(self, concept_seq, correct_seq, question_seq, n=1):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        use_question = encoder_config["use_question"]

        # concept_emb是融合了习题难度的concept，所以实际上可以看做question的embedding
        seqs_length = (correct_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, question_difficulty = self.embed_input({
            "concept_seq": concept_seq, "correct_seq": correct_seq, "question_seq": question_seq
        })
        z, q_scores, k_scores = self(concept_emb, interaction_emb, seqs_length)

        query = concept_emb[:, n - 1:, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        predict_score = self.out(torch.cat([query, latent], dim=-1)).squeeze(-1)

        if use_question:
            reg_loss = (question_difficulty ** 2).mean()
        else:
            reg_loss = 0.0

        return predict_score, z, concept_emb, reg_loss, (q_scores, k_scores)

    def get_predict_loss_(self, batch, loss_record):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        window = encoder_config["window"]

        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]

        # reg_loss实际上就是question difficulty embedding的二范数，和AKT一样，作为loss的一部分
        predict_logits, z, concept_emb, reg_loss, _ = self.predict(concept_seq, correct_seq, question_seq)
        loss_record.add_loss("reg loss", reg_loss.detach().cpu().item(), 1)
        weight_reg_loss = self.params["loss_config"]["reg loss"]
        reg_loss = reg_loss * weight_reg_loss

        ground_truth = correct_seq[correct_seq >= 0].float()
        predict_logits = predict_logits[correct_seq >= 0]
        # binary_cross_entropy_with_logits = Sigmoid + BCE loss，因此predict_logits是任意数
        predict_loss = F.binary_cross_entropy_with_logits(predict_logits, ground_truth, reduction="mean")

        for i in range(1, window):
            label = correct_seq[:, i:]
            query = concept_emb[:, i:, :]
            h = self.readout(z[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            predict_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )

        predict_loss /= window

        return predict_loss + reg_loss

    def get_cl_loss(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        dropout = encoder_config["dropout"]
        use_question = encoder_config["use_question"]
        use_hard_neg = encoder_config["use_hard_neg"]

        concept_seq = batch["concept_seq"]
        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]

        batch_size = correct_seq.size(0)
        seqs_length = (correct_seq >= 0).sum(dim=1)
        min_len = seqs_length.min().item()

        # augmentation
        concept_seq_ = concept_seq.clone()
        correct_seq_ = correct_seq.clone()

        question_seq_ = None
        if use_question:
            question_seq_ = question_seq.clone()

        # manipulate order
        for b in range(batch_size):
            idx = random.sample(
                range(seqs_length[b] - 1), max(1, int(seqs_length[b] * dropout))
            )
            for i in idx:
                concept_seq_[b, i], concept_seq_[b, i + 1] = concept_seq_[b, i + 1], concept_seq_[b, i]
                correct_seq_[b, i], correct_seq_[b, i + 1] = correct_seq_[b, i + 1], correct_seq_[b, i]
                if use_question:
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
        z_1 = self.get_z(batch)
        z_2 = self.get_z({"concept_seq": concept_seq_, "correct_seq": correct_seq_, "question_seq": question_seq_})
        input_ = self.sim(z_1[:, :min_len, :], z_2[:, :min_len, :])
        if use_hard_neg:
            z_3 = self.get_z({"concept_seq": concept_seq, "correct_seq": s_flip, "question_seq": question_seq})
            hard_neg = self.sim(z_1[:, :min_len, :], z_3[:, :min_len, :])
            input_ = torch.cat((input_, hard_neg), dim=1)
        target = (
            torch.arange(correct_seq.size(0))[:, None]
            .to(self.params["device"])
            .expand(-1, min_len)
        )
        cl_loss = F.cross_entropy(input_, target)

        return cl_loss

    def sim(self, z1, z2):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DTransformer"]
        temp = encoder_config["temp"]

        bs, seq_len, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seq_len, self.num_knowledge_prototype, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seq_len, self.num_knowledge_prototype, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)

        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / temp


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
