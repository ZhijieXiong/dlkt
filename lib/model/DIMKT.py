import numpy as np
import torch.nn as nn

from .BaseModel4CL import BaseModel4CL
from .Module.MLP import MLP4LLM_emb
from .util import *
from .loss_util import binary_entropy
from ..util.parse import concept2question_from_Q, question2concept_from_Q


class DIMKT(nn.Module, BaseModel4CL):
    model_name = "DIMKT"

    def __init__(self, params, objects):
        super(DIMKT, self).__init__()
        super(nn.Module, self).__init__(params, objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        dim_emb = encoder_config["dim_emb"]
        num_question_diff = encoder_config["num_question_diff"]
        num_concept_diff = encoder_config["num_concept_diff"]
        dropout = encoder_config["dropout"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        self.embed_question = self.get_embed_question()
        self.embed_concept = self.get_embed_concept()
        self.embed_question_diff = nn.Embedding(num_question_diff + 1, dim_emb)
        self.embed_concept_diff = nn.Embedding(num_concept_diff + 1, dim_emb)
        self.embed_correct = nn.Embedding(2, dim_emb)

        if use_LLM_emb4question:
            dim_LLM_emb = self.embed_question_difficulty.weight.shape[1]
            self.MLP4question = MLP4LLM_emb(dim_LLM_emb, dim_emb, 0.1)
        if use_LLM_emb4concept:
            dim_LLM_emb = self.embed_concept_variation.weight.shape[1]
            self.MLP4concept = MLP4LLM_emb(dim_LLM_emb, dim_emb, 0.1)

        self.generate_x_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.SDF_MLP1 = nn.Linear(dim_emb, dim_emb)
        self.SDF_MLP2 = nn.Linear(dim_emb, dim_emb)
        self.PKA_MLP1 = nn.Linear(2 * dim_emb, dim_emb)
        self.PKA_MLP2 = nn.Linear(2 * dim_emb, dim_emb)
        self.knowledge_indicator_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.dropout_layer = nn.Dropout(dropout)

        # 解析q table
        self.question2concept_list = question2concept_from_Q(objects["data"]["Q_table"])
        self.concept2question_list = concept2question_from_Q(objects["data"]["Q_table"])
        self.embed_question4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.question2concept_list,
                                                               self.concept2question_list)

    def get_embed_question(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        dim_emb = encoder_config["dim_emb"]
        num_question = encoder_config["num_question"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]

        if use_LLM_emb4question:
            LLM_question_embeddings = self.objects["data"]["LLM_question_embeddings"]
            all_embeddings = np.array([emb for emb in LLM_question_embeddings.values()])
            mean_embedding = all_embeddings.mean(axis=0).tolist()
            q_id2original_c_id = self.objects["data"]["q_id2original_c_id"]
            data_type = self.params["datasets_config"]["data_type"]

            embed_question = []
            if data_type == "only_question":
                pass
            else:
                for c_id in q_id2original_c_id:
                    if str(c_id) in LLM_question_embeddings.keys():
                        embed_question.append(LLM_question_embeddings[str(c_id)])
                    else:
                        embed_question.append(mean_embedding)
            embed_question = torch.tensor(embed_question, dtype=torch.float).to(self.params["device"])
            embed = nn.Embedding(num_question, embed_question.shape[1], _weight=embed_question)
            embed.weight.requires_grad = self.params["train_LLM_emb"]
        else:
            # 题目难度用一个embedding表示
            embed = nn.Embedding(num_question, dim_emb)

        return embed

    def get_embed_concept(self):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        dim_emb = encoder_config["dim_emb"]
        num_concept = encoder_config["num_concept"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        if use_LLM_emb4concept:
            LLM_concept_embeddings = self.objects["data"]["LLM_concept_embeddings"]
            all_embeddings = np.array([emb for emb in LLM_concept_embeddings.values()])
            mean_embedding = all_embeddings.mean(axis=0).tolist()
            c_id2original_c_id = self.objects["data"]["c_id2original_c_id"]
            data_type = self.params["datasets_config"]["data_type"]

            embed_concept = []
            if data_type == "only_question":
                pass
            else:
                for i in range(len(c_id2original_c_id)):
                    c_id = c_id2original_c_id[i]
                    if str(c_id) in LLM_concept_embeddings.keys():
                        embed_concept.append(LLM_concept_embeddings[str(c_id)])
                    else:
                        embed_concept.append(mean_embedding)

            embed_concept = torch.tensor(embed_concept, dtype=torch.float).to(self.params["device"])
            embed = nn.Embedding(embed_concept.shape[0], embed_concept.shape[1], _weight=embed_concept)
            embed.weight.requires_grad = self.params["train_LLM_emb"]
        else:
            embed = nn.Embedding(num_concept, dim_emb)

        return embed

    def get_concept_emb(self):
        return self.embed_concept.weight

    def forward(self, batch):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_question(batch["question_seq"])
        if use_LLM_emb4question:
            question_emb = self.MLP4question(question_emb)
        concept_emb = self.embed_concept(batch["concept_seq"])
        if use_LLM_emb4concept:
            concept_emb = self.MLP4concept(concept_emb)
        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])
        correct_emb = self.embed_correct(batch["correct_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len-1):
            input_x = torch.concat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.concat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.concat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.concat((
                question_emb[:, t+1],
                concept_emb[:, t+1],
                question_diff_emb[:, t+1],
                concept_diff_emb[:, t+1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        return y

    def get_latent(self, batch, use_emb_dropout=False, dropout=0.1):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]
        question_emb = self.embed_question(batch["question_seq"])
        if use_LLM_emb4question:
            question_emb = self.MLP4question(question_emb)
        concept_emb = self.embed_concept(batch["concept_seq"])
        if use_LLM_emb4concept:
            concept_emb = self.MLP4concept(concept_emb)
        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])
        correct_emb = self.embed_correct(batch["correct_seq"])
        if use_emb_dropout:
            question_emb = torch.dropout(question_emb, dropout, self.training)
            concept_emb = torch.dropout(concept_emb, dropout, self.training)
            question_diff_emb = torch.dropout(question_diff_emb, dropout, self.training)
            concept_diff_emb = torch.dropout(concept_diff_emb, dropout, self.training)
            correct_emb = torch.dropout(correct_emb, dropout, self.training)

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.concat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.concat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.concat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            latent[:, t + 1, :] = h
            h_pre = h

        return latent

    def get_latent_last(self, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent(batch, use_emb_dropout, dropout)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean(self, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent(batch, use_emb_dropout, dropout)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def set_emb4zero(self):
        """
        transfer head to tail use gaussian distribution
        :return:
        """
        head2tail_transfer_method = self.params["head2tail_transfer_method"]
        indices = []
        tail_qs_emb = []
        for z_q, head_qs in self.question_head4zero.items():
            head_question_indices = torch.tensor(head_qs).long().to(self.params["device"])
            head_qs_emb = self.embed_question(head_question_indices).detach().clone()
            if len(head_qs) == 0:
                continue
            indices.append(z_q)
            if head2tail_transfer_method == "mean_pool":
                tail_q_emb = head_qs_emb.mean(dim=0).unsqueeze(0)
            else:
                raise NotImplementedError()
            tail_qs_emb.append(tail_q_emb.float())
        indices = torch.tensor(indices)
        tail_qs_emb = torch.cat(tail_qs_emb, dim=0)

        embed_question = self.embed_question.weight.detach().clone()
        embed_question[indices] = tail_qs_emb
        self.embed_question4zero = nn.Embedding(embed_question.shape[0], embed_question.shape[1], _weight=embed_question)

    def get_predict_score4question_zero(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_question4zero(batch["question_seq"])
        if use_LLM_emb4question:
            question_emb = self.MLP4question(question_emb)
        concept_emb = self.embed_concept(batch["concept_seq"])
        if use_LLM_emb4concept:
            concept_emb = self.MLP4concept(concept_emb)
        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])
        correct_emb = self.embed_correct(batch["correct_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.concat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.concat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.concat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.concat((
                question_emb[:, t + 1],
                concept_emb[:, t + 1],
                question_diff_emb[:, t + 1],
                concept_diff_emb[:, t + 1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        predict_score = torch.masked_select(y[:, :-1], mask_bool_seq[:, 1:])
        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, :-1], mask_bool_seq[:, 1:])

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

    def forward_from_adv_data(self, dataset, batch):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]
        question_emb = dataset["embed_question"](batch["question_seq"])
        concept_emb = dataset["embed_concept"](batch["concept_seq"])
        question_diff_emb = dataset["embed_question_diff"](batch["question_diff_seq"])
        concept_diff_emb = dataset["embed_concept_diff"](batch["concept_diff_seq"])
        correct_emb = dataset["embed_correct"](batch["correct_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.concat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.concat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.concat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.concat((
                question_emb[:, t + 1],
                concept_emb[:, t + 1],
                question_diff_emb[:, t + 1],
                concept_diff_emb[:, t + 1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        return y

    def get_latent_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]
        question_emb = dataset["embed_question"](batch["question_seq"])
        concept_emb = dataset["embed_concept"](batch["concept_seq"])
        question_diff_emb = dataset["embed_question_diff"](batch["question_diff_seq"])
        concept_diff_emb = dataset["embed_concept_diff"](batch["concept_diff_seq"])
        correct_emb = dataset["embed_correct"](batch["correct_seq"])
        if use_emb_dropout:
            question_emb = torch.dropout(question_emb, dropout, self.training)
            concept_emb = torch.dropout(concept_emb, dropout, self.training)
            question_diff_emb = torch.dropout(question_diff_emb, dropout, self.training)
            concept_diff_emb = torch.dropout(concept_diff_emb, dropout, self.training)
            correct_emb = torch.dropout(correct_emb, dropout, self.training)

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.concat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.concat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.concat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            latent[:, t + 1, :] = h
            h_pre = h

        return latent

    def get_latent_last_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch, use_emb_dropout, dropout)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch, use_emb_dropout, dropout)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_predict_score_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward_from_adv_data(dataset, batch)
        predict_score = torch.masked_select(predict_score[:, :-1], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score_from_adv_data(dataset, batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        return predict_loss

    def max_entropy_adv_aug(self, dataset, batch, optimizer, loop_adv, eta, gamma):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        latent_ori = self.get_latent_from_adv_data(dataset, batch).detach().clone()
        latent_ori = latent_ori[mask_bool_seq]
        latent_ori.requires_grad_(False)
        adv_predict_loss = 0.
        adv_entropy = 0.
        adv_mse_loss = 0.
        for ite_max in range(loop_adv):
            predict_score = self.get_predict_score_from_adv_data(dataset, batch)
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
            question_seq = batch["question_seq"]
            question_difficulty_emb = dataset["embed_question_difficulty"](question_seq)
            rasch_loss = (question_difficulty_emb ** 2.).sum()
            predict_loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]
            entropy_loss = binary_entropy(predict_score)
            latent = self.get_latent_from_adv_data(dataset, batch)
            latent = latent[mask_bool_seq]
            latent_mse_loss = nn.functional.mse_loss(latent, latent_ori)

            if ite_max == (loop_adv - 1):
                adv_predict_loss += predict_loss.detach().cpu().item()
                adv_entropy += entropy_loss.detach().cpu().item()
                adv_mse_loss += latent_mse_loss.detach().cpu().item()
            loss = predict_loss + eta * entropy_loss - gamma * latent_mse_loss
            self.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

        return adv_predict_loss, adv_entropy, adv_mse_loss

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)[:, :-1]
