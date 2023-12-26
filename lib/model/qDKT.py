from sklearn.mixture import GaussianMixture

from .BaseModel4CL import BaseModel4CL
from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .Module.MLP import MLP4LLM_emb
from .loss_util import *
from .util import *
from ..util.parse import concept2question_from_Q, question2concept_from_Q


class qDKT(nn.Module, BaseModel4CL):
    model_name = "qDKT"

    def __init__(self, params, objects):
        super(qDKT, self).__init__()
        super(nn.Module, self).__init__(params, objects)

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            self.embed_layer.parse_Q_table()

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

        self.predict_layer = PredictorLayer(self.params, self.objects)

        # 解析q table
        if params["transfer_head2zero"]:
            self.question2concept_list = question2concept_from_Q(objects["data"]["Q_table"])
            self.concept2question_list = concept2question_from_Q(objects["data"]["Q_table"])
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.question2concept_list,
                                                               self.concept2question_list)
            self.embed_question4zero = None

        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        embed_config = self.params["models_config"]["kt_model"]["kt_embed_layer"]
        if use_LLM_emb4question:
            dim_LLM_emb = self.embed_layer.embed_question.weight.shape[1]
            dim_question = embed_config["question"][1]
            self.MLP4question = MLP4LLM_emb(dim_LLM_emb, dim_question, 0.1)
        if use_LLM_emb4concept:
            dim_LLM_emb = self.embed_layer.embed_concept.weight.shape[1]
            dim_concept = embed_config["question"][1]
            self.MLP4concept = MLP4LLM_emb(dim_LLM_emb, dim_concept, 0.1)

    def get_concept_emb(self):
        return self.embed_layer.get_emb_all("concept")

    def get_qc_emb4single_concept(self, batch):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        if (not use_LLM_emb4question) and (not use_LLM_emb4concept):
            concept_question_emb = self.embed_layer.get_emb_concatenated(("concept", "question"),
                                                                         (concept_seq, question_seq))
        else:
            concept_emb = self.embed_layer.get_emb("concept", concept_seq)
            question_emb = self.embed_layer.get_emb("question", question_seq)
            if use_LLM_emb4concept:
                concept_emb = self.MLP4concept(concept_emb)
            if use_LLM_emb4question:
                question_emb = self.MLP4question(question_emb)
            concept_question_emb = torch.cat((concept_emb, question_emb), dim=-1)

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], concept_fusion="mean")

    def forward(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        data_type = self.params["datasets_config"]["data_type"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"][:, 1:], penultimate=False)

        return predict_score[mask4last.bool()]

    def get_latent(self, batch, use_emb_dropout=False, dropout=0.1):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        data_type = self.params["datasets_config"]["data_type"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        if use_emb_dropout:
            qc_emb = torch.dropout(qc_emb, dropout, self.training)
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

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
        transfer head to tail use gaussian distribution
        :return:
        """
        data_type = self.params["datasets_config"]["data_type"]
        head2tail_transfer_method = self.params["head2tail_transfer_method"]
        indices = []
        tail_qs_emb = []
        for z_q, head_qs in self.question_head4zero.items():
            head_question_indices = torch.tensor(head_qs).long().to(self.params["device"])
            head_qs_emb = self.embed_layer.get_emb("question", head_question_indices).detach().clone()
            if len(head_qs) == 0:
                continue
            indices.append(z_q)
            if head2tail_transfer_method == "gaussian_fit":
                # zero shot question的emb用同一知识点下的其它习题emb拟合高斯分布然后从里面采样
                if len(head_qs_emb) > 100:
                    head_qs_emb = head_qs_emb.detach().cpu().numpy()
                    if data_type == "only_question":
                        # 多知识点数据集
                        n_com = 2
                    else:
                        # 单知识点数据集
                        n_com = 1
                    gmm = GaussianMixture(n_components=n_com, random_state=self.params["seed"])
                    gmm.fit(head_qs_emb)
                    gmm_samples = gmm.sample(1)
                    tail_q_emb = torch.from_numpy(gmm_samples[0][0]).unsqueeze(0).to(self.params["device"])
                else:
                    tail_q_emb = head_qs_emb.mean(dim=0).unsqueeze(0)
            elif head2tail_transfer_method == "mean_pool":
                tail_q_emb = head_qs_emb.mean(dim=0).unsqueeze(0)
            else:
                raise NotImplementedError()
            tail_qs_emb.append(tail_q_emb.float())
        indices = torch.tensor(indices)
        tail_qs_emb = torch.cat(tail_qs_emb, dim=0)

        embed_question = self.embed_layer.embed_question.weight.detach().clone()
        embed_question[indices] = tail_qs_emb
        self.embed_question4zero = nn.Embedding(embed_question.shape[0], embed_question.shape[1], _weight=embed_question)

    def get_predict_score4question_zero(self, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        data_type = self.params["datasets_config"]["data_type"]
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]
        batch_size = correct_seq.shape[0]

        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)

        if data_type == "only_question":
            # todo: 这里是错误的，需要改正
            qc_emb = self.embed_layer4zero.get_emb_question_with_concept_fused(batch["question_seq"], "mean")
        else:
            concept_emb = self.embed_layer.get_emb("concept", batch["concept_seq"])
            question_emb = self.embed_question4zero(batch["question_seq"])
            if use_LLM_emb4concept:
                concept_emb = self.MLP4concept(concept_emb)
            if use_LLM_emb4question:
                question_emb = self.MLP4question(question_emb)
            qc_emb = torch.cat((concept_emb, question_emb), dim=-1)

        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

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

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

    def forward_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = dataset["embed_layer"].get_emb_concatenated(("concept", "question"), (concept_seq, question_seq))
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_latent_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = dataset["embed_layer"].get_emb_concatenated(("concept", "question"), (concept_seq, question_seq))
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        return latent

    def get_latent_last_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"], penultimate=False)
        latent_last = latent[torch.where(mask4last == 1)]

        return latent_last

    def get_latent_mean_from_adv_data(self, dataset, batch, use_emb_dropout=False, dropout=0.1):
        latent = self.get_latent_from_adv_data(dataset, batch)
        mask_seq = batch["mask_seq"]
        latent_mean = (latent * mask_seq.unsqueeze(-1)).sum(1) / mask_seq.sum(-1).unsqueeze(-1)

        return latent_mean

    def get_predict_score_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward_from_adv_data(dataset, batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss_from_adv_data(self, dataset, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.get_predict_score_from_adv_data(dataset, batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        rasch_loss = self.get_rasch_loss(batch)
        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch_loss"]

        return loss

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
