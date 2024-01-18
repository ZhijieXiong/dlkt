import numpy as np
import torch.nn as nn

from .BaseModel4CL import BaseModel4CL
from .Module.MLP import MLP4LLM_emb
from .util import *
from .loss_util import binary_entropy
from ..util.parse import concept2question_from_Q, question2concept_from_Q
from ..util.data import context2batch


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
        self.embed_question_diff = nn.Embedding(num_question_diff, dim_emb)
        self.embed_concept_diff = nn.Embedding(num_concept_diff, dim_emb)
        self.embed_correct = nn.Embedding(2, dim_emb)

        if use_LLM_emb4question:
            dim_LLM_emb = self.embed_question.weight.shape[1]
            self.MLP4question = MLP4LLM_emb(dim_LLM_emb, dim_emb, 0.1)
        if use_LLM_emb4concept:
            dim_LLM_emb = self.embed_concept.weight.shape[1]
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
        self.embed_question_diff4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.question2concept_list,
                                                               self.concept2question_list)
            # question_no_head_qs = []
            # for z_q, head_qs in self.question_head4zero.items():
            #     if len(head_qs) == 0:
            #         question_no_head_qs.append(z_q)
            #
            # # 看一下有哪些习题是训练集中没出现过，并且相同知识点下的head习题也为0
            # self.objects["data"]["question_no_head_qs"] = question_no_head_qs

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

    def get_concept_emb_all(self):
        return self.embed_concept.weight.detach().clone()

    def get_target_question_emb(self, target_question):
        return self.embed_question(target_question)

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
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
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
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
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
        if use_emb_dropout:
            latent_last = torch.dropout(latent_last, dropout, self.training)

        return latent_last

    def get_latent_mean(self, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent(batch, use_emb_dropout, dropout)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)
        if use_emb_dropout:
            latent_mean = torch.dropout(latent_mean, dropout, self.training)

        return latent_mean

    def set_emb4zero(self):
        """
        transfer knowledge of head question to tail question
        :return:
        """
        head2tail_transfer_method = self.params["head2tail_transfer_method"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]
        num_question = encoder_config["num_question"]
        num_question_diff = encoder_config["num_question_diff"]
        question_difficulty = self.objects["dimkt"]["question_difficulty"]

        indices = []
        tail_qs_emb = []
        tail_qs_diff_emb = []
        embed_question_diff = self.embed_question_diff.weight[num_question_diff].detach().clone()
        embed_question_diff = embed_question_diff.unsqueeze(0).repeat(num_question, 1)
        # 哪些习题是训练集中没出现过，并且相同知识点下的head习题也为0
        self.objects["data"]["question_no_head_qs"] = []
        for z_q, head_qs in self.question_head4zero.items():
            head_question_indices = torch.tensor(head_qs).long().to(self.params["device"])
            head_qs_emb = self.embed_question(head_question_indices).detach().clone()

            head_qs_diff = list(map(lambda q_id: question_difficulty.get(q_id, num_question_diff), head_qs))
            head_qs_diff = list(filter(lambda q_diff: q_diff != num_question_diff, head_qs_diff))
            head_question_diff_indices = torch.tensor(head_qs_diff).long().to(self.params["device"])
            head_qs_diff_emb = self.embed_question_diff(head_question_diff_indices).detach().clone()

            indices.append(z_q)
            if len(head_qs) == 0:
                self.objects["data"]["question_no_head_qs"].append(z_q)
                tail_q_emb = self.embed_question.weight.mean(dim=0).detach().clone().unsqueeze(0)
                tail_q_diff_emb = self.embed_question_diff.weight.mean(dim=0).detach().clone().unsqueeze(0)
                tail_qs_emb.append(tail_q_emb.float())
                tail_qs_diff_emb.append(tail_q_diff_emb.float())
                continue

            if head2tail_transfer_method == "mean_pool":
                tail_q_emb = head_qs_emb.mean(dim=0).unsqueeze(0)
                tail_q_diff_emb = head_qs_diff_emb.mean(dim=0).unsqueeze(0)
            elif head2tail_transfer_method == "max_pool":
                tail_q_emb = torch.max(head_qs_emb, dim=1)[0].unsqueeze(0)
                tail_q_diff_emb = torch.max(head_qs_diff_emb, dim=1)[0].unsqueeze(0)
            elif head2tail_transfer_method == "zero_pad":
                tail_q_emb = torch.zeros((1, head_qs_emb.shape[-1])).to(self.params["device"])
                tail_q_diff_emb = torch.zeros((1, head_qs_diff_emb.shape[-1])).to(self.params["device"])
            elif head2tail_transfer_method == "most_popular":
                # 因为是排过序的，最前面的就是频率最高的
                tail_q_emb = head_qs_emb[0:1]
                tail_q_diff_emb = head_qs_diff_emb[0:1]
            else:
                raise NotImplementedError()

            tail_qs_emb.append(tail_q_emb.float())
            tail_qs_diff_emb.append(tail_q_diff_emb.float())

        not_zero_qs = list(set(range(num_question)) - set(self.question_head4zero.keys()))
        for not_zero_q_id in not_zero_qs:
            q_diff_id = question_difficulty.get(not_zero_q_id, num_question_diff)
            embed_question_diff[not_zero_q_id] = self.embed_question_diff.weight[q_diff_id].detach().clone()

        indices = torch.tensor(indices)
        tail_qs_emb = torch.cat(tail_qs_emb, dim=0)
        tail_qs_diff_emb = torch.cat(tail_qs_diff_emb, dim=0)

        embed_question = self.embed_question.weight.detach().clone()
        embed_question[indices] = tail_qs_emb
        # 如果在习题难度层面上迁移知识到零频率，反而效果变差
        embed_question_diff[indices] = tail_qs_diff_emb
        self.embed_question4zero = nn.Embedding(num_question, embed_question.shape[1], _weight=embed_question)
        self.embed_question_diff4zero = nn.Embedding(num_question, embed_question_diff.shape[1], _weight=embed_question_diff)

    def get_predict_score4question_zero(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
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

        # 聚合的zero question emb只用于最后的预测层
        question_emb4zero = self.embed_question4zero(batch["question_seq"])
        # 对于diff如果使用聚合的，只会使效果变差
        # question_diff_emb4zero = self.embed_question_diff4zero(batch["question_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
                question_emb4zero[:, t + 1],
                concept_emb[:, t + 1],
                # 对于diff如果使用聚合的，只会使效果变差
                # question_diff_emb4zero[:, t],
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

    def get_predict_score4long_tail(self, batch, seq_branch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        use_transfer4seq = self.params["other"]["mutual_enhance4long_tail"]["use_transfer4seq"]
        beta = self.params["other"]["mutual_enhance4long_tail"]["beta4transfer_seq"]
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

        for t in range(seq_len - 1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            if use_transfer4seq and t <= 10:
                h_transferred = (beta * h + seq_branch.get_latent_transferred(h)) / (1 + beta)
            else:
                h_transferred = h
            input_x_next = torch.cat((
                question_emb[:, t + 1],
                concept_emb[:, t + 1],
                question_diff_emb[:, t + 1],
                concept_diff_emb[:, t + 1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h_transferred, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        predict_score = torch.masked_select(y[:, :-1], mask_bool_seq[:, 1:])

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

    def get_predict_enhance_loss(self, batch, loss_record=None):
        enhance_method = self.params["other"]["output_enhance"]["enhance_method"]
        weight_enhance_loss1 = self.params["loss_config"]["enhance loss 1"]
        weight_enhance_loss2 = self.params["loss_config"]["enhance loss 2"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_question(batch["question_seq"])
        if enhance_method == 0 or enhance_method == 1:
            question_emb_easier = self.embed_question(batch["question_easier_seq"])
            question_emb_harder = self.embed_question(batch["question_harder_seq"])
        else:
            question_emb_easier = None
            question_emb_harder = None
        if enhance_method == 0 or enhance_method == 2:
            question_emb4zero = self.embed_question(batch["question_zero_shot_seq"])
        else:
            question_emb4zero = None
        if use_LLM_emb4question:
            question_emb = self.MLP4question(question_emb)
            question_emb_easier = self.MLP4question(question_emb_easier)
            question_emb_harder = self.MLP4question(question_emb_harder)
            question_emb4zero = self.MLP4question(question_emb4zero)

        concept_emb = self.embed_concept(batch["concept_seq"])
        if enhance_method == 0 or enhance_method == 1:
            concept_emb_easier = self.embed_concept(batch["concept_easier_seq"])
            concept_emb_harder = self.embed_concept(batch["concept_harder_seq"])
        else:
            concept_emb_easier = None
            concept_emb_harder = None
        if enhance_method == 0 or enhance_method == 2:
            concept_emb4zero = self.embed_concept(batch["concept_zero_shot_seq"])
        else:
            concept_emb4zero = None
        if use_LLM_emb4concept:
            concept_emb = self.MLP4concept(concept_emb)
            concept_emb_easier = self.MLP4concept(concept_emb_easier)
            concept_emb_harder = self.MLP4concept(concept_emb_harder)
            concept_emb4zero = self.MLP4concept(concept_emb4zero)

        question_diff_emb = self.embed_question_diff(batch["question_diff_seq"])
        if enhance_method == 0 or enhance_method == 1:
            question_diff_emb_easier = self.embed_question_diff(batch["question_easier_diff_seq"])
            question_diff_emb_harder = self.embed_question_diff(batch["question_harder_diff_seq"])
        else:
            question_diff_emb_easier = None
            question_diff_emb_harder = None
        if enhance_method == 0 or enhance_method == 2:
            question_diff_emb4zero = self.embed_question_diff(batch["question_zero_shot_diff_seq"])
        else:
            question_diff_emb4zero = None

        concept_diff_emb = self.embed_concept_diff(batch["concept_diff_seq"])
        if enhance_method == 0 or enhance_method == 1:
            concept_diff_emb_easier = self.embed_concept_diff(batch["concept_easier_diff_seq"])
            concept_diff_emb_harder = self.embed_concept_diff(batch["concept_harder_diff_seq"])
        else:
            concept_diff_emb_easier = None
            concept_diff_emb_harder = None
        if enhance_method == 0 or enhance_method == 2:
            concept_diff_emb4zero = self.embed_concept_diff(batch["concept_zero_shot_diff_seq"])
        else:
            concept_diff_emb4zero = None

        correct_emb = self.embed_correct(batch["correct_seq"])
        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        predict_score_all = torch.zeros(batch_size, seq_len).to(self.params["device"])
        predict_score_easier_all = torch.zeros(batch_size, seq_len).to(self.params["device"])
        predict_score_harder_all = torch.zeros(batch_size, seq_len).to(self.params["device"])
        predict_score_last4zero = torch.zeros(batch_size, seq_len).to(self.params["device"])
        predict_score_current4zero = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len - 1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h_current = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka

            input_x_next = torch.cat((
                question_emb[:, t + 1],
                concept_emb[:, t + 1],
                question_diff_emb[:, t + 1],
                concept_diff_emb[:, t + 1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            predict_score_all[:, t] = torch.sigmoid(torch.sum(x_next * h_current, dim=-1))

            if enhance_method == 0 or enhance_method == 1:
                # enhance method 1: easier
                input_x_next_easier = torch.cat((
                    question_emb_easier[:, t + 1],
                    concept_emb_easier[:, t + 1],
                    question_diff_emb_easier[:, t + 1],
                    concept_diff_emb_easier[:, t + 1]
                ), dim=1)
                x_next_easier = self.generate_x_MLP(input_x_next_easier)
                predict_score_easier_all[:, t] = torch.sigmoid(torch.sum(x_next_easier * h_current, dim=-1))

                # enhance method 1: harder
                input_x_next_harder = torch.cat((
                    question_emb_harder[:, t + 1],
                    concept_emb_harder[:, t + 1],
                    question_diff_emb_harder[:, t + 1],
                    concept_diff_emb_harder[:, t + 1]
                ), dim=1)
                x_next_harder = self.generate_x_MLP(input_x_next_harder)
                predict_score_harder_all[:, t] = torch.sigmoid(torch.sum(x_next_harder * h_current, dim=-1))

            if enhance_method == 0 or enhance_method == 2:
                # enhance method 2
                input_x_next4zero = torch.cat((
                    question_emb4zero[:, t],
                    concept_emb4zero[:, t],
                    question_diff_emb4zero[:, t],
                    concept_diff_emb4zero[:, t]
                ), dim=1)
                x_next4zero = self.generate_x_MLP(input_x_next4zero)

                predict_score_last4zero[:, t] = torch.sigmoid(torch.sum(x_next4zero * h_pre, dim=-1))
                predict_score_current4zero[:, t] = torch.sigmoid(torch.sum(x_next4zero * h_current, dim=-1))

            latent[:, t + 1, :] = h_current
            h_pre = h_current

        loss = 0.
        # 预测损失
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score_all[:, :-1], mask_bool_seq[:, 1:])
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss = loss + predict_loss

        # enhance method 1: 对于easier和harder习题的损失
        if enhance_method == 0 or enhance_method == 1:
            mask_bool_seq_easier = torch.ne(batch["mask_easier_seq"], 0)
            mask_bool_seq_harder = torch.ne(batch["mask_harder_seq"], 0)
            weight_easier = torch.masked_select(batch["weight_easier_seq"][:, 1:], mask_bool_seq_easier[:, 1:])
            weight_harder = torch.masked_select(batch["weight_harder_seq"][:, 1:], mask_bool_seq_harder[:, 1:])

            predict_score_diff1 = predict_score_easier_all - predict_score_all
            predict_score_diff1 = torch.masked_select(predict_score_diff1[:, :-1], mask_bool_seq_easier[:, 1:])
            predict_score_diff2 = predict_score_all - predict_score_harder_all
            predict_score_diff2 = torch.masked_select(predict_score_diff2[:, :-1], mask_bool_seq_harder[:, 1:])
            enhance_loss_easier = -torch.min(torch.zeros_like(predict_score_diff1).to(self.params["device"]),
                                             predict_score_diff1)
            enhance_loss_easier = enhance_loss_easier * weight_easier
            enhance_loss_harder = -torch.min(torch.zeros_like(predict_score_diff2).to(self.params["device"]),
                                             predict_score_diff2)
            enhance_loss_harder = enhance_loss_harder * weight_harder
            enhance_loss1 = enhance_loss_easier.mean() + enhance_loss_harder.mean()

            if loss_record is not None:
                loss_record.add_loss("enhance loss 1", enhance_loss1.detach().cpu().item(), 1)
            loss = loss + enhance_loss1 * weight_enhance_loss1

        # enhance loss2: 对于zero shot的习题，用单调理论约束
        if enhance_method == 0 or enhance_method == 2:
            mask_zero_shot_seq = torch.ne(batch["mask_zero_shot_seq"], 0)
            predict_score_diff4zero = predict_score_current4zero - predict_score_last4zero
            predict_score_diff4zero = torch.masked_select(predict_score_diff4zero[:, :-1], mask_zero_shot_seq[:, :-1])
            enhance_loss2 = -torch.min(torch.zeros_like(predict_score_diff4zero).to(self.params["device"]),
                                       predict_score_diff4zero).mean()

            if loss_record is not None:
                loss_record.add_loss("enhance loss 2", enhance_loss2.detach().cpu().item(), 1)
            loss = loss + enhance_loss2 * weight_enhance_loss2

        return loss

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
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correct_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
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
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correct_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
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

    def get_question_transferred_1_branch(self, batch_question, question_branch):
        dataset_train = self.objects["mutual_enhance4long_tail"]["dataset_train"]
        device = self.params["device"]
        question_context = self.objects["mutual_enhance4long_tail"]["question_context"]

        context_batch = []
        idx_list = [0]
        idx = 0
        tail_question_list = []
        for i, q_id in enumerate(batch_question[0].cpu().tolist()):
            if not question_context.get(q_id, False):
                continue

            tail_question_list.append(q_id)
            context_list = question_context[q_id]
            context_batch += context_list
            idx += len(context_list)
            idx_list.append(idx)

        if len(context_batch) == 0:
            return

        context_batch = context2batch(dataset_train, context_batch, device)
        latent = self.get_latent_last(context_batch)

        question_emb_transferred = []
        for i in range(len(tail_question_list)):
            mean_context = latent[idx_list[i]: idx_list[i + 1]]
            mean_context = question_branch.get_question_emb_transferred(mean_context.mean(0), True)
            question_emb_transferred.append(mean_context)

        question_emb_transferred = torch.stack(question_emb_transferred)
        return question_emb_transferred, tail_question_list

    def get_question_transferred_2_branch(self, batch_question, question_branch):
        dataset_train = self.objects["mutual_enhance4long_tail"]["dataset_train"]
        device = self.params["device"]
        question_context = self.objects["mutual_enhance4long_tail"]["question_context"]
        dim_latent = self.params["other"]["mutual_enhance4long_tail"]["dim_latent"]
        dim_question = self.params["other"]["mutual_enhance4long_tail"]["dim_question"]

        right_context_batch = []
        wrong_context_batch = []
        right_idx_list = [0]
        wrong_idx_list = [0]
        idx4right = 0
        idx4wrong = 0
        tail_question_list = []
        for i, q_id in enumerate(batch_question[0].cpu().tolist()):
            if not question_context.get(q_id, False):
                continue

            tail_question_list.append(q_id)
            right_context_list = []
            wrong_context_list = []
            for q_context in question_context[q_id]:
                if q_context["correct"] == 1:
                    right_context_list.append(q_context)
                else:
                    wrong_context_list.append(q_context)
            right_context_batch += right_context_list
            wrong_context_batch += wrong_context_list

            l1 = len(right_context_list)
            if l1 >= 1:
                idx4right += l1
                right_idx_list.append(idx4right)
            else:
                right_idx_list.append(right_idx_list[-1])

            l2 = len(wrong_context_list)
            if l2 >= 1:
                idx4wrong += l2
                wrong_idx_list.append(idx4wrong)
            else:
                wrong_idx_list.append(wrong_idx_list[-1])

        if len(right_context_batch) == 0 and len(wrong_context_batch) == 0:
            return

        if len(right_context_batch) > 0:
            right_context_batch = context2batch(dataset_train, right_context_batch, device)
            latent_right = self.get_latent_last(right_context_batch)
        else:
            latent_right = torch.empty((0, dim_latent))

        if len(wrong_context_batch) > 0:
            wrong_context_batch = context2batch(dataset_train, wrong_context_batch, device)
            latent_wrong = self.get_latent_last(wrong_context_batch)
        else:
            latent_wrong = torch.empty((0, dim_latent))

        question_emb_transferred = []
        for i in range(len(tail_question_list)):
            mean_right_context = latent_right[right_idx_list[i]: right_idx_list[i + 1]]
            mean_wrong_context = latent_wrong[wrong_idx_list[i]: wrong_idx_list[i + 1]]
            num_right = len(mean_right_context)
            num_wrong = len(mean_wrong_context)
            coef_right = num_right / (num_right + num_wrong)
            coef_wrong = num_wrong / (num_right + num_wrong)
            if num_right == 0:
                mean_right_context = torch.zeros(dim_question).float().to(self.params["device"])
            else:
                mean_right_context = question_branch.get_question_emb_transferred(mean_right_context.mean(0), True)
            if num_wrong == 0:
                mean_wrong_context = torch.zeros(dim_question).float().to(self.params["device"])
            else:
                mean_wrong_context = question_branch.get_question_emb_transferred(mean_wrong_context.mean(0), False)
            question_emb_transferred.append(coef_right * mean_right_context + coef_wrong * mean_wrong_context)

        question_emb_transferred = torch.stack(question_emb_transferred)
        return question_emb_transferred, tail_question_list

    def update_tail_question(self, batch_question, question_branch):
        gamma = self.params["other"]["mutual_enhance4long_tail"]["gamma4transfer_question"]
        two_branch4question_transfer = self.params["other"]["mutual_enhance4long_tail"]["two_branch4question_transfer"]

        if two_branch4question_transfer:
            question_emb_transferred, tail_question_list = \
                self.get_question_transferred_2_branch(batch_question, question_branch)
        else:
            question_emb_transferred, tail_question_list = \
                self.get_question_transferred_1_branch(batch_question, question_branch)

        question_emb = self.get_target_question_emb(torch.tensor(tail_question_list).long().to(self.params["device"]))
        # 这里官方代码实现和论文上写的不一样
        self.embed_question.weight.data[tail_question_list] = (question_emb_transferred + gamma * question_emb) / (1 + gamma)

    def freeze_emb(self):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        self.embed_question.weight.requires_grad = False
        self.embed_concept.weight.requires_grad = False
        self.embed_question_diff.weight.requires_grad = False
        self.embed_concept_diff.weight.requires_grad = False
        self.embed_correct.weight.requires_grad = False

        if use_LLM_emb4question:
            for param in self.MLP4question.parameters():
                param.requires_grad = False
        if use_LLM_emb4concept:
            for param in self.MLP4concept.parameters():
                param.requires_grad = False

    def get_question_emb_all(self):
        return self.embed_question.weight.detach().clone()
