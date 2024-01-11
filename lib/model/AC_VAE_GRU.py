from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.AC_VAE import *


class AC_VAE_GRU(nn.Module):
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = KTEmbedLayer(self.params, self.objects)
        data_type = self.params["datasets_config"]["data_type"]
        if data_type == "only_question":
            self.embed_layer.parse_Q_table()
        self.embed_dropout = nn.Dropout(0.5)

        rnn_config = self.params["models_config"]["kt_model"]["rnn_layer"]
        dim_concept = rnn_config["dim_concept"]
        dim_question = rnn_config["dim_question"]
        dim_correct = rnn_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_rnn = rnn_config["dim_rnn"]
        rnn_type = rnn_config["rnn_type"]
        num_rnn_layer = rnn_config["num_rnn_layer"]
        if rnn_type == "rnn":
            self.rnn_layer = nn.RNN(dim_emb, dim_rnn, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.rnn_layer = nn.LSTM(dim_emb, dim_rnn, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.rnn_layer = nn.GRU(dim_emb, dim_rnn, batch_first=True, num_layers=num_rnn_layer)

        encoder_type = self.params["models_config"]["kt_model"]["encoder_layer"]["type"]
        if encoder_type == 'fc':
            self.encoder_layer = FCEncoder(params)
        elif encoder_type == 'fc_cnn':
            self.encoder_layer = FCEncoderCNN(params)
        elif encoder_type == 'fc_no_res':
            self.encoder_layer = FCEncoderNoRes(params)
        else:
            raise NotImplementedError()

        self.decoder_layer = Decoder(params)

    def get_qc_emb4single_concept(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        concept_question_emb = self.embed_layer.get_emb_concatenated(("concept", "question"),
                                                                     (concept_seq, question_seq))

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], fusion_type="mean")

    def forward(self, batch):
        dim_correct = self.params["models_config"]["kt_model"]["rnn_layer"]["dim_correct"]
        data_type = self.params["datasets_config"]["data_type"]
        correct_seq = batch["correct_seq"]

        batch_size = correct_seq.shape[0]
        correct_emb = correct_seq.reshape(-1, 1).repeat(1, dim_correct).reshape(batch_size, -1, dim_correct)
        qc_emb = self.get_qc_emb4only_question(batch) if data_type == "only_question" else (
            self.get_qc_emb4single_concept(batch))
        interaction_emb = torch.cat((qc_emb[:, :-1], correct_emb[:, :-1]), dim=2)
        interaction_emb_real = interaction_emb

        self.rnn_layer.flatten_parameters()
        rnn_out, _ = self.rnn_layer(self.embed_dropout(interaction_emb))
        latent_inferred = self.encoder_layer(rnn_out)
        predict_score = self.decoder_layer(latent_inferred, qc_emb[:, 1:])

        return predict_score, interaction_emb_real, latent_inferred

    def get_latent(self, batch):
        pass

    def get_latent_last(self, batch):
        pass

    def get_latent_mean(self, batch):
        pass

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score, _, _ = self.forward(batch)

        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])

        return predict_score

    def get_loss_stage1(self, batch, loss_record=None, anneal_value=1):
        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        num_seq = batch["mask_seq"].shape[0]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])

        predict_score, interaction_emb_real, latent_inferred = self.forward(batch)

        # 预测损失，对应VAE中的重构损失，这一部分AVB和VAE一样
        predict_score = torch.masked_select(predict_score, mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        contrastive_discriminator = self.objects["models"]["contrastive_discriminator"]
        adversary_discriminator = self.objects["models"]["adversary_discriminator"]

        # kl_loss：公式（10）中后一项（即AdvDiscriminator的输出），对应VAE中的KL项
        use_anneal = self.params["other"]["adv_contrastive_vae"]["ues_anneal"]
        if use_anneal:
            weight_kl = anneal_value
        else:
            weight_kl = self.params["loss_config"]["adv loss"]
        t_joint = adversary_discriminator(interaction_emb_real, latent_inferred, batch["mask_seq"])
        t_joint = torch.mean(t_joint, dim=-1)
        kl_loss = torch.sum(t_joint) / float(num_seq)
        if loss_record is not None:
            loss_record.add_loss("adv loss stage1", kl_loss.detach().cpu().item(), 1)
        kl_loss = kl_loss * weight_kl

        # 对比损失
        t_joint = contrastive_discriminator(interaction_emb_real, latent_inferred, batch["mask_seq"])
        z_shuffled = torch.cat([latent_inferred[1:], latent_inferred[:1]], dim=0)
        t_shuffled = contrastive_discriminator(interaction_emb_real, z_shuffled, batch["mask_seq"])
        Ej = -F.softplus(-t_joint)
        Em = F.softplus(t_shuffled)
        GLOBAL = (Em - Ej) * batch["mask_seq"][:, :-1].float()
        cl_loss = torch.sum(GLOBAL) / float(num_seq)
        if loss_record is not None:
            loss_record.add_loss("cl loss", cl_loss.detach().cpu().item(), 1)
        cl_loss = cl_loss * self.params["loss_config"]["cl loss"]

        loss = predict_loss + kl_loss + cl_loss

        return loss

    def get_loss_stage2(self, batch, loss_record=None):
        num_seq = batch["mask_seq"].shape[0]
        adversary_discriminator = self.objects["models"]["adversary_discriminator"]

        # 公式7：对抗AdvDiscriminator
        _, interaction_emb_real, latent_inferred = self.forward(batch)
        prior = torch.randn_like(latent_inferred)
        term_a = torch.log(torch.sigmoid(adversary_discriminator(interaction_emb_real.detach(), latent_inferred.detach(), batch["mask_seq"])) + 1e-9)
        term_b = torch.log(1.0 - torch.sigmoid(adversary_discriminator(interaction_emb_real.detach(), prior, batch["mask_seq"])) + 1e-9)
        PRIOR = -torch.mean(term_a + term_b, dim=-1) * batch["mask_seq"][:, :-1].float()
        adv_kl_loss = torch.sum(PRIOR) / float(num_seq)
        if loss_record is not None:
            loss_record.add_loss("adv loss stage2", adv_kl_loss.detach().cpu().item(), 1)

        return adv_kl_loss

