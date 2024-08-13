from copy import deepcopy
from .BaseModel4CL import BaseModel4CL
from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.PredictorLayer import PredictorLayer
from .Module.MLP import MLP4LLM_emb
from .loss_util import *
from .util import *
from ..util.data import context2batch


class qDKT(nn.Module, BaseModel4CL):
    model_name = "qDKT"

    def __init__(self, params, objects):
        super(qDKT, self).__init__()
        super(nn.Module, self).__init__(params, objects)

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

        self.predict_layer = PredictorLayer(self.params, self.objects)

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

        # 解析q table
        self.embed_question4zero = None
        self.question_head4zero = None
        if self.objects["data"].get("train_data_statics", False):
            self.question_head4zero = parse_question_zero_shot(self.objects["data"]["train_data_statics"],
                                                               self.objects["data"]["question2concept"],
                                                               self.objects["data"]["concept2question"])

    def get_concept_emb_all(self):
        return self.embed_layer.get_emb_all("concept")

    def get_target_question_emb(self, target_question):
        return self.embed_layer.get_emb("question", target_question)

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
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], fusion_type="mean")

    def forward4question_evaluate(self, batch):
        # 直接输出的是每个序列最后一个时刻的预测分数
        predict_score = self.forward(batch)
        # 只保留mask每行的最后一个1
        mask4last = get_mask4last_or_penultimate(batch["mask_seq"][:, 1:], penultimate=False)

        return predict_score[mask4last.bool()]

    def get_predict_score_seq_len_minus1(self, batch):
        return self.forward(batch)

    # ------------------------------------------------------base--------------------------------------------------------

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

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_score4mix_up_sample(self, batch):
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

        batch4mix_up = deepcopy(batch)
        batch4mix_up["question_seq"] = batch["question_seq4mix_up"]
        qc_emb4mix_up = self.get_qc_emb4only_question(batch4mix_up)
        predict_layer_input = torch.cat((latent, (qc_emb[:, 1:] + qc_emb4mix_up[:, 1:]) / 2), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        mask_bool_seq = torch.ne(batch["mask_seq4mix_up"], 0)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        adv_bias_aug = self.params["other"].get("adv_bias_aug", None)
        use_sample_weight = True
        if adv_bias_aug is not None:
            adv_bias_aug_ablation = self.params["other"]["adv_bias_aug"]["ablation"]
            if adv_bias_aug_ablation == 9:
                use_sample_weight = False
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        if self.params.get("use_sample_weight", False) and use_sample_weight:
            weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(),
                                                              ground_truth.double(),
                                                              weight=weight)
        else:
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        # use_mix_up = self.params.get("use_mix_up", False)
        # if use_mix_up:
        #     weight4mix_up_sample = self.params["weight4mix_up_sample"]
        #     predict_score4mix_up = self.get_predict_score4mix_up_sample(batch)
        #     mask_bool_seq4mix_up = torch.ne(batch["mask_seq4mix_up"], 0)
        #     ground_truth4mix_up = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq4mix_up[:, 1:])
        #     if self.params.get("use_sample_weight", False) and use_sample_weight:
        #         weight4mix_up = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq4mix_up[:, 1:])
        #         predict_loss4mix_up = nn.functional.binary_cross_entropy(predict_score4mix_up.double(),
        #                                                                  ground_truth4mix_up.double(),
        #                                                                  weight=weight4mix_up)
        #     else:
        #         predict_loss4mix_up = nn.functional.binary_cross_entropy(
        #             predict_score4mix_up.double(), ground_truth4mix_up.double()
        #         )
        #
        #     if loss_record is not None:
        #         num_sample4mix_up = torch.sum(batch["mask_seq4mix_up"][:, 1:]).item()
        #         loss_record.add_loss(
        #             "mix up predict loss",
        #             predict_loss4mix_up.detach().cpu().item() * num_sample4mix_up,
        #             num_sample4mix_up
        #         )
        #     predict_loss += predict_loss4mix_up * weight4mix_up_sample

        return predict_loss

    def get_GCE_loss(self, batch, q=0.7):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        # predict_score全是为1的概率
        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        # predict_score_gt为1或者为0的概率，如果标签为1，则是为1的概率，如果标签是0，则是为0的概率
        predict_score_gt = predict_score * ground_truth + (1 - predict_score) * (1 - ground_truth)
        weight = (predict_score_gt.detach() ** q) * q
        GCE_loss = nn.functional.binary_cross_entropy(
            predict_score.double(), ground_truth.double(), weight=weight
        )

        return GCE_loss

    def get_predict_loss_per_sample(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss_per_sample = nn.functional.binary_cross_entropy(
            predict_score.double(), ground_truth.double(), reduction="none"
        )

        return predict_loss_per_sample

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

    def get_predict_score4target_question(self, latent, question_seq, concept_seq=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]
        data_type = self.params["datasets_config"]["data_type"]
        num_latent = len(latent)
        num_question = len(question_seq)

        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question({"question_seq": question_seq})
        else:
            qc_emb = self.get_qc_emb4single_concept({
                "question_seq": question_seq,
                "concept_seq": concept_seq
            })
        predict_layer_input = torch.cat((
            latent.repeat(1, num_question).view(num_latent, num_question, dim_latent),
            qc_emb.view(1, num_question, dim_concept + dim_question).repeat(num_latent, 1, 1)
        ), dim=-1)

        return self.predict_layer(predict_layer_input).squeeze(dim=-1)

    # ------------------------------------------------SRS---------------------------------------------------------------
    def get_predict_score_srs(self, batch):
        data_type = self.params["datasets_config"]["data_type"]

        batch_size = len(batch["seq_len"])
        idx1 = torch.arange(batch_size).long().to(self.params["device"])
        idx2 = batch["seq_len"] - 1
        latent = self.get_latent(batch)[idx1, idx2]
        if data_type == "only_question":
            target_qc_emb = self.embed_layer.get_emb_question_with_concept_fused(batch["target_question"], "mean")
        else:
            target_q_emb = self.embed_layer.get_emb("question", batch["target_question"])
            target_c_emb = self.embed_layer.get_emb("concept", batch["target_concept"])
            target_qc_emb = torch.cat((target_c_emb, target_q_emb), dim=1)
        predict_layer_input = torch.cat((latent, target_qc_emb), dim=1)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_loss_srs(self, batch, loss_record=None):
        ground_truth = batch["target_correct"]
        predict_score = self.get_predict_score_srs(batch)
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = len(batch["seq_len"])
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def get_predict_score4all_question_srs(self, batch):
        data_type = self.params["datasets_config"]["data_type"]
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_concept = encoder_config["dim_concept"]
        dim_question = encoder_config["dim_question"]
        dim_latent = encoder_config["dim_latent"]

        batch_size = len(batch["seq_len"])
        idx1 = torch.arange(batch_size).long().to(self.params["device"])
        idx2 = batch["seq_len"] - 1
        latent = self.get_latent(batch)[idx1, idx2]

        Q_table = self.objects["data"]["Q_table"]
        num_question = Q_table.shape[0]
        question_all = torch.arange(num_question).long().to(self.params["device"])
        if data_type == "only_question":
            all_qc_emb = self.embed_layer.get_emb_question_with_concept_fused(question_all, "mean")
            predict_layer_input = torch.cat((
                latent.repeat(1, num_question).view(batch_size, num_question, dim_latent),
                all_qc_emb.view(1, num_question, dim_concept+dim_question).repeat(batch_size, 1, 1)
            ), dim=-1)
        elif data_type == "multi_concept":
            raise NotImplementedError()
        else:
            question_all_emb = self.embed_layer.get_emb("question", question_all)
            concept_all = torch.from_numpy(np.nonzero(Q_table)[1]).long().to(self.params["device"])
            concept_all_emb = self.embed_layer.get_emb("concept", concept_all)

            predict_layer_input = torch.cat((
                latent.repeat(1, num_question).view(batch_size, num_question, dim_latent),
                concept_all_emb.view(1, num_question, dim_concept).repeat(batch_size, 1, 1),
                question_all_emb.view(1, num_question, dim_question).repeat(batch_size, 1, 1)
            ), dim=-1)

        predict_score = self.predict_layer(predict_layer_input)

        return predict_score.squeeze(dim=-1)

    def get_dro_loss(self, batch, loss_record=None):
        loss = 0.
        ground_truth = batch["target_correct"]
        question_next = batch["target_question"]

        predict_score = self.get_predict_score_srs(batch)
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        if loss_record is not None:
            num_sample = len(batch["seq_len"])
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)
        loss += predict_loss

        model_output = self.get_predict_score4all_question_srs(batch)
        propensity_score = self.objects["dro"]["propensity"]
        beta = self.params["other"]["dro"]["beta"]
        alpha = self.params["loss_config"]["dro loss"]

        # 做对和做错对应的index
        correct_indices = torch.nonzero(ground_truth).view(-1)
        incorrect_indices = torch.nonzero(ground_truth - 1).view(-1)
        target_pos = torch.gather(question_next, 0, correct_indices)
        target_neg = torch.gather(question_next, 0, incorrect_indices)
        # nominal distribution下让样本接近0和1的损失
        loss_mu2zero = torch.mul(model_output * model_output, propensity_score)
        loss_mu2one = torch.mul((model_output - 1) * (model_output - 1), propensity_score)
        # 让做对的知识点|习题靠近1，远离0
        pos_scores_dro = loss_mu2zero[correct_indices, target_pos]
        pos_loss_dro = loss_mu2one[correct_indices, target_pos]
        inner_dro_pos = torch.exp((pos_loss_dro / beta)) - torch.exp((pos_scores_dro / beta))
        # 让做错的知识点|习题靠近0，远离1
        neg_scores_dro = loss_mu2one[incorrect_indices, target_neg]
        neg_loss_dro = loss_mu2zero[incorrect_indices, target_neg]
        inner_dro_neg = torch.exp((neg_loss_dro / beta)) - torch.exp((neg_scores_dro / beta))
        # 每个时刻，如果做对，所有的知识点|习题都靠近0；如果做错，所有的知识点|习题都靠近1（最差的分布）
        inner_dro_all_pos = torch.sum(
            torch.exp((torch.mul(model_output * model_output, propensity_score) / beta)), 1
        )[correct_indices]
        inner_dro_all_neg = torch.sum(
            torch.exp((torch.mul((model_output - 1) * (model_output - 1), propensity_score) / beta)), 1
        )[incorrect_indices]
        # 对应项相加（做对的和做错的）
        inner_dro = torch.cat((inner_dro_all_pos + inner_dro_pos, inner_dro_all_neg + inner_dro_neg))
        loss_DRO = torch.mean(torch.log(inner_dro + 1e-24))

        if loss_record is not None:
            loss_record.add_loss("dro loss", loss_DRO.detach().cpu().item(), 1)

        loss += alpha * loss_DRO

        return loss

    def get_cl_loss_srs(self, batch):
        batch_aug0 = {
            "question_seq": batch["question_seq_aug_0"],
            "correct_seq": batch["correct_seq_aug_0"]
        }
        batch_aug1 = {
            "question_seq": batch["question_seq_aug_1"],
            "correct_seq": batch["correct_seq_aug_1"]
        }
        if "concept_seq" in batch.keys():
            batch_aug0["concept_seq"] = batch["concept_seq_aug_0"]
            batch_aug1["concept_seq"] = batch["concept_seq_aug_1"]

        batch_size = batch["seq_len"].shape[0]
        batch_idx = range(batch_size)
        latent_aug0 = self.get_latent(batch_aug0)[batch_idx, batch["seq_len_aug_0"]]
        latent_aug1 = self.get_latent(batch_aug1)[batch_idx, batch["seq_len_aug_1"]]
        labels = torch.arange(batch_size).long().to(self.params["device"])
        temp = self.params["other"]["instance_cl"]["temp"]
        cos_sim = torch.cosine_similarity(latent_aug0.unsqueeze(1), latent_aug1.unsqueeze(0), dim=-1) / temp
        cl_loss = nn.functional.cross_entropy(cos_sim, labels)

        return cl_loss

    # -------------------------------transfer head item to zero shot item-----------------------------------------------

    def set_emb4zero(self):
        """
        transfer head to tail use gaussian distribution
        :return:
        """
        head2tail_transfer_method = self.params["transfer_head2zero"]["transfer_method"]
        indices = []
        tail_qs_emb = []
        for z_q, head_qs in self.question_head4zero.items():
            head_question_indices = torch.tensor(head_qs).long().to(self.params["device"])
            head_qs_emb = self.embed_layer.get_emb("question", head_question_indices).detach().clone()
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

    # --------------------------------------------output enhance--------------------------------------------------------

    def get_predict_enhance_loss(self, batch, loss_record=None):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        enhance_method = self.params["other"]["output_enhance"]["enhance_method"]
        weight_enhance_loss1 = self.params["loss_config"]["enhance loss 1"]
        weight_enhance_loss2 = self.params["loss_config"]["enhance loss 2"]
        dim_correct = encoder_config["dim_correct"]
        correct_seq = batch["correct_seq"]
        data_type = self.params["datasets_config"]["data_type"]

        loss = 0.
        # 预测损失
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        latent_current = latent[:, :-1]
        latent_next = latent[:, 1:]

        predict_layer_input = torch.cat((latent_current, qc_emb[:, 1:]), dim=2)
        predict_score_all = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score_all, mask_bool_seq[:, 1:])

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
            batch_easier = {
                "question_seq": batch["question_easier_seq"],
            }
            batch_harder = {
                "question_seq": batch["question_harder_seq"],
            }
            if "concept_seq" in batch.keys():
                batch_easier["concept_seq"] = batch["concept_easier_seq"]
                batch_harder["concept_seq"] = batch["concept_harder_seq"]

            if data_type == "only_question":
                qc_emb_easier = self.get_qc_emb4only_question(batch_easier)
                qc_emb_harder = self.get_qc_emb4only_question(batch_harder)
            else:
                qc_emb_easier = self.get_qc_emb4single_concept(batch_easier)
                qc_emb_harder = self.get_qc_emb4only_question(batch_harder)

            predict_input_easier = torch.cat((latent_current, qc_emb_easier[:, 1:]), dim=2)
            predict_input_harder = torch.cat((latent_current, qc_emb_harder[:, 1:]), dim=2)
            predict_score_easier_all = self.predict_layer(predict_input_easier).squeeze(dim=-1)
            predict_score_harder_all = self.predict_layer(predict_input_harder).squeeze(dim=-1)

            predict_score_diff1 = torch.masked_select(predict_score_easier_all - predict_score_all, mask_bool_seq_easier[:, 1:])
            predict_score_diff2 = torch.masked_select(predict_score_all - predict_score_harder_all, mask_bool_seq_harder[:, 1:])
            enhance_loss_easier = -torch.min(torch.zeros_like(predict_score_diff1).to(self.params["device"]), predict_score_diff1)
            enhance_loss_easier = enhance_loss_easier * weight_easier
            enhance_loss_harder = -torch.min(torch.zeros_like(predict_score_diff2).to(self.params["device"]), predict_score_diff2)
            enhance_loss_harder = enhance_loss_harder * weight_harder

            enhance_loss1 = (enhance_loss_easier.mean() + enhance_loss_harder.mean()) / 2

            if loss_record is not None:
                loss_record.add_loss("enhance loss 1", enhance_loss1.detach().cpu().item(), 1)
            loss = loss + enhance_loss1 * weight_enhance_loss1

        # enhance loss2: 对于zero shot的习题，用单调理论约束
        if enhance_method == 0 or enhance_method == 2:
            mask_zero_shot_seq = torch.ne(batch["mask_zero_shot_seq"], 0)
            batch_zero_shot = {
                "question_seq": batch["question_zero_shot_seq"]
            }
            if "concept_seq" in batch.keys():
                batch_zero_shot["concept_seq"] = batch["concept_zero_shot_seq"]
            if data_type == "only_question":
                qc_emb_zero_shot = self.get_qc_emb4only_question(batch_zero_shot)
            else:
                qc_emb_zero_shot = self.get_qc_emb4single_concept(batch_zero_shot)
            predict_input_current4zero = torch.cat((latent_current, qc_emb_zero_shot[:, :-1]), dim=2)
            predict_input_next4zero = torch.cat((latent_next, qc_emb_zero_shot[:, :-1]), dim=2)
            predict_score_current4zero = self.predict_layer(predict_input_current4zero).squeeze(dim=-1)
            predict_score_next4zero = self.predict_layer(predict_input_next4zero).squeeze(dim=-1)
            predict_score_diff4zero = torch.masked_select(predict_score_next4zero - predict_score_current4zero,
                                                          mask_zero_shot_seq[:, :-1])
            enhance_loss2 = -torch.min(torch.zeros_like(predict_score_diff4zero).to(self.params["device"]), predict_score_diff4zero).mean()

            if loss_record is not None:
                loss_record.add_loss("enhance loss 2", enhance_loss2.detach().cpu().item(), 1)
            loss = loss + enhance_loss2 * weight_enhance_loss2

        return loss

    # --------------------------------------------------ME-ADA----------------------------------------------------------

    def forward_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        data_type = self.params["datasets_config"]["data_type"]
        dim_correct = encoder_config["dim_correct"]

        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            question_emb = dataset["embed_layer"].get_emb("question", question_seq)
            concept_emb = dataset["embed_layer"].get_concept_fused_emb(question_seq, fusion_type="mean")
            qc_emb = torch.cat((concept_emb, question_emb), dim=-1)
        else:
            concept_seq = batch["concept_seq"]
            qc_emb = dataset["embed_layer"].get_emb_concatenated(("concept", "question"), (concept_seq, question_seq))
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_latent_from_adv_data(self, dataset, batch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        data_type = self.params["datasets_config"]["data_type"]
        dim_correct = encoder_config["dim_correct"]

        correct_seq = batch["correct_seq"]
        question_seq = batch["question_seq"]
        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            question_emb = dataset["embed_layer"].get_emb("question", question_seq)
            concept_emb = dataset["embed_layer"].get_concept_fused_emb(question_seq, fusion_type="mean")
            qc_emb = torch.cat((concept_emb, question_emb), dim=-1)
        else:
            concept_seq = batch["concept_seq"]
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

    def get_predict_score_from_adv_data(self, dataset, batch, mask4adv=None):
        predict_score = self.forward_from_adv_data(dataset, batch)
        if mask4adv is None:
            mask_bool_seq = torch.ne(batch["mask_seq"], 0)
            predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])
        else:
            predict_score = torch.masked_select(predict_score, mask4adv)

        return predict_score

    # def get_predict_score_from_adv_data4mix_up(self, dataset, batch):
    #     encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
    #     data_type = self.params["datasets_config"]["data_type"]
    #     dim_correct = encoder_config["dim_correct"]
    #
    #     correct_seq = batch["correct_seq"]
    #     question_seq = batch["question_seq"]
    #     batch_size = correct_seq.shape[0]
    #     correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
    #     if data_type == "only_question":
    #         question_emb = dataset["embed_layer"].get_emb("question", question_seq)
    #         concept_emb = dataset["embed_layer"].get_concept_fused_emb(question_seq, fusion_type="mean")
    #         qc_emb = torch.cat((concept_emb, question_emb), dim=-1)
    #     else:
    #         concept_seq = batch["concept_seq"]
    #         qc_emb = dataset["embed_layer"].get_emb_concatenated(("concept", "question"), (concept_seq, question_seq))
    #     interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)
    #
    #     self.encoder_layer.flatten_parameters()
    #     latent, _ = self.encoder_layer(interaction_emb)
    #
    #     question_emb = dataset["embed_layer"].get_emb("question", batch["question_seq4mix_up"])
    #     concept_emb = dataset["embed_layer"].get_concept_fused_emb(batch["question_seq4mix_up"], fusion_type="mean")
    #     qc_emb4mix_up = torch.cat((concept_emb, question_emb), dim=-1)
    #
    #     predict_layer_input = torch.cat((latent, (qc_emb[:, 1:] + qc_emb4mix_up[:, 1:]) / 2), dim=2)
    #     predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
    #
    #     pass

    def get_predict_loss_from_adv_data(self, dataset, batch, mask4adv=None):
        ablation = self.params["other"]["adv_bias_aug"]["ablation"]
        if mask4adv is None:
            predict_score = self.get_predict_score_from_adv_data(dataset, batch)
            mask_bool_seq = torch.ne(batch["mask_seq"], 0)
            ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
            if self.params.get("use_sample_weight", False) and ablation != 9:
                weight = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq[:, 1:])
                predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double(), weight=weight)
            else:
                predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        else:
            predict_score = self.get_predict_score_from_adv_data(dataset, batch, mask4adv)
            ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask4adv)
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        # use_mix_up = self.params.get("use_mix_up", False)
        # if use_mix_up:
        #     weight4mix_up_sample = self.params["weight4mix_up_sample"]
        #     predict_score4mix_up = self.get_predict_score4mix_up_sample(batch)
        #     mask_bool_seq4mix_up = torch.ne(batch["mask_seq4mix_up"], 0)
        #     ground_truth4mix_up = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq4mix_up[:, 1:])
        #     if self.params.get("use_sample_weight", False):
        #         weight4mix_up = torch.masked_select(batch["weight_seq"][:, 1:], mask_bool_seq4mix_up[:, 1:])
        #         predict_loss4mix_up = nn.functional.binary_cross_entropy(predict_score4mix_up.double(),
        #                                                                  ground_truth4mix_up.double(),
        #                                                                  weight=weight4mix_up)
        #     else:
        #         predict_loss4mix_up = nn.functional.binary_cross_entropy(
        #             predict_score4mix_up.double(), ground_truth4mix_up.double()
        #         )
        #
        #     predict_loss += predict_loss4mix_up * weight4mix_up_sample

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

    def adv_bias_aug(self, dataset, batch, optimizer, loop_adv, eta, gamma, mask4gen=None):
        ablation = self.params["other"]["adv_bias_aug"]["ablation"]
        if mask4gen is None:
            mask4gen = torch.ne(batch["mask_seq"][:, 1:], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask4gen)

        latent_ori = self.get_latent_from_adv_data(dataset, batch).detach().clone()
        latent_ori = latent_ori[:, 1:][mask4gen]
        latent_ori.requires_grad_(False)
        adv_predict_loss = 0.
        adv_entropy = 0.
        adv_mse_loss = 0.
        for ite_max in range(loop_adv):
            predict_score = self.forward_from_adv_data(dataset, batch)
            predict_score = torch.masked_select(predict_score, mask4gen)
            if ablation == 9:
                weight = torch.masked_select(batch["weight_seq"][:, 1:], mask4gen)
                weight = 2 - weight
                predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double(), weight=weight)
            else:
                predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
            # entropy_loss = binary_entropy(predict_score)
            latent = self.get_latent_from_adv_data(dataset, batch)
            latent = latent[:, 1:][mask4gen]
            latent_mse_loss = nn.functional.mse_loss(latent, latent_ori)

            if ite_max == (loop_adv - 1):
                adv_predict_loss += predict_loss.detach().cpu().item()
                # adv_entropy += entropy_loss.detach().cpu().item()
                adv_mse_loss += latent_mse_loss.detach().cpu().item()
            # loss = predict_loss + eta * entropy_loss - gamma * latent_mse_loss
            loss = predict_loss - gamma * latent_mse_loss
            self.zero_grad()
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

        return adv_predict_loss, adv_mse_loss, adv_entropy

    def get_interaction_emb(self, batch):
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
        interaction_emb = torch.cat((qc_emb, correct_emb), dim=2)

        return interaction_emb
    # ----------------------------------------------------MELT----------------------------------------------------------

    def get_predict_score4long_tail(self, batch, seq_branch):
        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]["qDKT"]
        dim_correct = encoder_config["dim_correct"]
        data_type = self.params["datasets_config"]["data_type"]
        use_transfer4seq = self.params["other"]["mutual_enhance4long_tail"]["use_transfer4seq"]
        beta = self.params["other"]["mutual_enhance4long_tail"]["beta4transfer_seq"]

        correct_seq = batch["correct_seq"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        batch_size, seq_len = correct_seq.shape[0], correct_seq.shape[1]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        if data_type == "only_question":
            qc_emb = self.get_qc_emb4only_question(batch)
        else:
            qc_emb = self.get_qc_emb4single_concept(batch)
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        if use_transfer4seq:
            # t=10时刻以后的经过seq branch再送入predict layer
            mask4select1 = torch.zeros(seq_len - 1).to(self.params["device"])
            mask4select1[:, :10] = 1
            mask4select1 = mask4select1.bool().repeat(batch_size, 1)
            mask4select2 = torch.zeros(seq_len - 1).to(self.params["device"])
            mask4select2[:, 10:] = 1
            mask4select2 = mask4select2.bool().repeat(batch_size, 1)
            latent_transferred = (beta * latent + seq_branch.get_latent_transferred(latent)) / (1 + beta)
            latent = latent * mask4select1 + latent_transferred * mask4select2

        predict_layer_input = torch.cat((latent, qc_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

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
        self.embed_layer.embed_question.weight.data[tail_question_list] = \
            (question_emb_transferred + gamma * question_emb) / (1 + gamma)

    def freeze_emb(self):
        use_LLM_emb4question = self.params["use_LLM_emb4question"]
        use_LLM_emb4concept = self.params["use_LLM_emb4concept"]

        self.embed_question.weight.requires_grad = False
        self.embed_concept.weight.requires_grad = False

        if use_LLM_emb4question:
            for param in self.MLP4question.parameters():
                param.requires_grad = False
        if use_LLM_emb4concept:
            for param in self.MLP4concept.parameters():
                param.requires_grad = False
