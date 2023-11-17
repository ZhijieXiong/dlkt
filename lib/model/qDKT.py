import torch.optim as optim

from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .loss_util import *
from .util import *


class qDKT(nn.Module):
    def __init__(self, params, objects):
        super(qDKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        dim_correct = encoder_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.encoder_layer = self.encoder_layer

        self.predict_layer = PredictorLayer(self.params, self.objects)

    def get_concept_emb(self):
        return self.embed_layer.get_emb_all("concept")

    def get_qc_emb(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        concept_question_emb = (
            self.embed_layer.get_emb_concatenated(("concept", "question"), (concept_seq, question_seq)))

        return concept_question_emb

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = self.get_qc_emb(batch)
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_latent(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = self.get_qc_emb(batch)
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        return latent

    def get_duo_cl_loss(self, batch):
        batch_ori = {
            "concept_seq": batch["concept_seq"],
            "question_seq": batch["question_seq"],
            "correct_seq": batch["correct_seq"],
            "mask_seq": batch["mask_seq"]
        }
        latent_ori = self.get_latent(batch_ori)
        mask4last_ori = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_ori = latent_ori[torch.where(mask4last_ori == 1)]

        batch_aug = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        latent_aug = self.get_latent(batch_aug)
        mask4last_aug = get_mask4last_or_penultimate(batch_aug["mask_seq"], penultimate=False)
        latent_aug = latent_aug[torch.where(mask4last_aug == 1)]

        batch_hard_neg = {
            "concept_seq": batch["concept_seq_hard_neg"],
            "question_seq": batch["question_seq_hard_neg"],
            "correct_seq": batch["correct_seq_hard_neg"],
            "mask_seq": batch["mask_seq_hard_neg"]
        }
        latent_hard_neg = self.get_latent(batch_hard_neg)
        mask4last_hard_neg = get_mask4last_or_penultimate(batch_hard_neg["mask_seq"], penultimate=False)
        latent_hard_neg = latent_hard_neg[torch.where(mask4last_hard_neg == 1)]

        temp = self.params["other"]["duo"]["temp"]
        cl_loss = duo_info_nce(latent_ori, latent_aug, temp, sim_type="cos", z_hard_neg=latent_hard_neg)

        return cl_loss

    def get_instance_cl_loss_cl4kt(self, batch):
        batch_ori = {
            "concept_seq": batch["concept_seq"],
            "question_seq": batch["question_seq"],
            "correct_seq": batch["correct_seq"],
            "mask_seq": batch["mask_seq"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }
        batch_aug2 = {
            "concept_seq": batch["concept_seq_aug_2"],
            "question_seq": batch["question_seq_aug_2"],
            "correct_seq": batch["correct_seq_aug_2"],
            "mask_seq": batch["mask_seq_aug_2"]
        }
        pass

    def get_instance_cl_loss_our(self, batch):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        batch_aug1 = {
            "concept_seq": batch["concept_seq_aug_1"],
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"],
            "mask_seq": batch["mask_seq_aug_1"]
        }

        latent_aug0 = self.get_latent(batch_aug0)
        mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)
        latent_aug0_last = latent_aug0[torch.where(mask4last_aug0 == 1)]

        latent_aug1 = self.get_latent(batch_aug1)
        mask4last_aug1 = get_mask4last_or_penultimate(batch_aug1["mask_seq"], penultimate=False)
        latent_aug1_last = latent_aug1[torch.where(mask4last_aug1 == 1)]

        bs = latent_aug0.shape[0]
        seq_len = latent_aug0.shape[1]
        m = (torch.eye(bs) == 0)

        # 将另一增强序列的每个时刻都作为一个neg
        neg_all = latent_aug1.repeat(bs, 1, 1).reshape(bs, bs, seq_len, -1)[m].reshape(bs, bs-1, seq_len, -1)
        mask_bool4neg = torch.ne(batch["mask_seq_aug_1"].repeat(bs, 1).reshape(bs, bs, -1)[m].reshape(bs, bs-1, -1), 0)

        temp = self.params["other"]["duo"]["temp"]
        cos_sim_list = []
        for i in range(bs):
            anchor = latent_aug0_last[i]
            pos = latent_aug1_last[i]
            neg = neg_all[i][:, 1:][mask_bool4neg[i][:, 1:]]
            sim_i = nn.functional.cosine_similarity(anchor, torch.cat((pos.unsqueeze(dim=0), neg), dim=0)) / temp
            cos_sim_list.append(sim_i.unsqueeze(dim=0))

        labels = torch.tensor([0]).long().to(self.params["device"])
        cl_loss = 0.
        for i in range(bs):
            cos_sim = cos_sim_list[i]
            cl_loss = cl_loss + nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_instance_cl_loss_our_adv(self, batch, dataset_adv):
        batch_aug0 = {
            "concept_seq": batch["concept_seq_aug_0"],
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"],
            "mask_seq": batch["mask_seq_aug_0"]
        }
        latent_aug0 = self.get_latent(batch_aug0)[:, 1:]
        mask4last_aug0 = get_mask4last_or_penultimate(batch_aug0["mask_seq"], penultimate=False)[:, 1:]
        latent_aug0_last = latent_aug0[torch.where(mask4last_aug0 == 1)]

        seq_ids = batch["seq_id"]
        emb_seq_aug1 = dataset_adv["emb_seq"][seq_ids.to("cpu")].to(self.params["device"])
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        concept_emb = emb_seq_aug1[:, 1:, :dim_concept]
        question_emb = emb_seq_aug1[:, 1:, dim_concept:(dim_concept + dim_question)]
        _, latent_aug1 = self.forward_from_input_emb(emb_seq_aug1[:, :-1], concept_emb, question_emb)
        mask4last_aug1 = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)[:, 1:]
        latent_aug1_last = latent_aug1[torch.where(mask4last_aug1 == 1)]

        bs = latent_aug0.shape[0]
        seq_len = latent_aug0.shape[1]
        m = (torch.eye(bs) == 0)

        # 将另一增强序列的每个时刻都作为一个neg
        neg_all = latent_aug1.repeat(bs, 1, 1).reshape(bs, bs, seq_len, -1)[m].reshape(bs, bs - 1, seq_len, -1)
        mask_bool4neg = (
            torch.ne(batch["mask_seq"][:, 1:].repeat(bs, 1).reshape(bs, bs, -1)[m].reshape(bs, bs - 1, -1), 0))

        temp = self.params["other"]["duo"]["temp"]
        cos_sim_list = []
        for i in range(bs):
            anchor = latent_aug0_last[i]
            pos = latent_aug1_last[i]
            neg = neg_all[i][:, 1:][mask_bool4neg[i][:, 1:]]
            sim_i = nn.functional.cosine_similarity(anchor, torch.cat((pos.unsqueeze(dim=0), neg), dim=0)) / temp
            cos_sim_list.append(sim_i.unsqueeze(dim=0))

        labels = torch.tensor([0]).long().to(self.params["device"])
        cl_loss = 0.
        for i in range(bs):
            cos_sim = cos_sim_list[i]
            cl_loss = cl_loss + nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    def get_predict_loss(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        return loss

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_input_emb(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = self.get_qc_emb(batch)
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        return interaction_emb

    def forward_from_input_emb(self, input_emb, concept_emb, question_emb):
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(input_emb)

        predict_layer_input = torch.cat((latent, concept_emb, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score, latent

    def get_max_entropy_adv_aug_emb(self, batch, adv_learning_rate, loop_adv, eta, gamma):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]

        correct_seq = batch["correct_seq"]
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        mask4penultimate = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=True)
        ground_truth = correct_seq[torch.where(mask4last == 1)]

        latent = self.get_latent(batch)
        latent_penultimate = latent[torch.where(mask4penultimate == 1)].detach().clone()

        inputs_max = self.get_input_emb(batch).detach().clone()
        latent_penultimate.requires_grad_(False)
        inputs_max.requires_grad_(True)
        optimizer = optim.SGD(params=[inputs_max], lr=adv_learning_rate)
        adv_predict_loss = 0.
        adv_entropy = 0.
        adv_mse_loss = 0.
        for ite_max in range(loop_adv):
            concept_emb = inputs_max[:, 1:, :dim_concept]
            question_emb = inputs_max[:, 1:, dim_concept:(dim_concept + dim_question)]
            predict_score, latent = self.forward_from_input_emb(inputs_max[:, :-1], concept_emb, question_emb)
            predict_score = predict_score[torch.where(mask4last[:, 1:] == 1)]
            adv_pred_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
            entropy_loss = binary_entropy(predict_score)
            latent_mse_loss = (
                nn.functional.mse_loss(latent[torch.where(mask4penultimate[:, :-1] == 1)], latent_penultimate))

            if ite_max == (loop_adv - 1):
                adv_predict_loss += adv_pred_loss.detach().cpu().item()
                adv_entropy += entropy_loss.detach().cpu().item()
                adv_mse_loss += latent_mse_loss.detach().cpu().item()
            loss = adv_pred_loss + eta * entropy_loss - gamma * latent_mse_loss
            self.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

        return inputs_max, adv_predict_loss, adv_entropy, adv_mse_loss