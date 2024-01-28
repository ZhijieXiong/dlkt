from torch import nn

from .Module.KTEmbedLayer import KTEmbedLayer
from .util import *
from .loss_util import binary_entropy


class LPKT(nn.Module):
    model_name = "LPKT"

    def __init__(self, params, objects):
        super(LPKT, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
        num_question = encoder_config["num_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_e = encoder_config["dim_e"]
        dim_k = encoder_config["dim_k"]
        dropout = encoder_config["dropout"]

        # 3600 sec: 1 hour, 43200 min: 1 month
        self.at_embed = nn.Embedding(3600 + 1, dim_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(43200 + 1, dim_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(num_question + 1, dim_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        self.linear_1 = nn.Linear(dim_correct + dim_e + dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(dim_e + dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------base--------------------------------------------------------

    def forward(self, batch):
        question_seq = batch["question_seq"]
        use_time_seq = batch["use_time_seq"]
        interval_time_seq = batch["interval_time_seq"]
        correct_seq = batch["correct_seq"]

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["LPKT"]
        num_concept = encoder_config["num_concept"]
        dim_correct = encoder_config["dim_correct"]
        dim_k = encoder_config["dim_k"]
        q_matrix = self.objects["LPKT"]["q_matrix"]

        batch_size, seq_len = question_seq.size(0), question_seq.size(1)
        e_embed_data = self.e_embed(question_seq)
        at_embed_data = self.at_embed(use_time_seq)
        it_embed_data = self.it_embed(interval_time_seq)
        correct_seq = correct_seq.view(-1, 1).repeat(1, dim_correct).view(batch_size, -1, dim_correct)
        h_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_k)).repeat(batch_size, 1, 1).to(self.params["device"])
        h_tilde_pre = None
        all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, correct_seq), 2))
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])

        pred = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(0, seq_len - 1):
            e = question_seq[:, t]
            # q_e: (bs, 1, n_skill)
            q_e = q_matrix[e].view(batch_size, 1, -1)
            it = it_embed_data[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, dim_k)
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, dim_k),
                it.repeat(1, n_skill).view(batch_size, -1, dim_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = q_matrix[question_seq[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, dim_k)
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / dim_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred

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

    # ----------------------------------------------------ME-ADA--------------------------------------------------------

    def forward_from_adv_data(self, dataset, batch):
        dim_emb = self.params["models_config"]["kt_model"]["encoder_layer"]["DIMKT"]["dim_emb"]
        data_type = self.params["datasets_config"]["data_type"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = dataset["embed_question"](batch["question_seq"])
        question_diff_emb = dataset["embed_question_diff"](batch["question_diff_seq"])
        if data_type == "only_question":
            diff_fuse_table = self.objects["dimkt"]["diff_fuse_table"]
            concept_diff_emb = KTEmbedLayer.other_fused_emb(
                dataset["embed_concept_diff"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                diff_fuse_table,
                fusion_type="mean"
            )
        else:
            concept_emb = dataset["embed_concept"](batch["concept_seq"])
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
        data_type = self.params["datasets_config"]["data_type"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = dataset["embed_question"](batch["question_seq"])
        question_diff_emb = dataset["embed_question_diff"](batch["question_diff_seq"])
        if data_type == "only_question":
            concept_emb = KTEmbedLayer.concept_fused_emb(
                dataset["embed_concept"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                fusion_type="mean"
            )
            diff_fuse_table = self.objects["dimkt"]["diff_fuse_table"]
            concept_diff_emb = KTEmbedLayer.other_fused_emb(
                dataset["embed_concept_diff"],
                self.objects["data"]["q2c_table"],
                self.objects["data"]["q2c_mask_table"],
                batch["question_seq"],
                diff_fuse_table,
                fusion_type="mean"
            )
        else:
            concept_emb = dataset["embed_concept"](batch["concept_seq"])
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