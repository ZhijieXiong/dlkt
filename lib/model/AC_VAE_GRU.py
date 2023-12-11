from .Module.KTEmbedLayer import KTEmbedLayer
from .Module.AC_VAE import *


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

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
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], concept_fusion="mean")

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
        decoder_out, predict_score = self.decoder(latent_inferred)

        return decoder_out, interaction_emb_real, latent_inferred, predict_score

    def get_score(self, batch):
        pass


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, params):
        super(ContrastiveDiscriminator, self).__init__()

        self.params = params
        rnn_config = self.params["models_config"]["kt_model"]["rnn_layer"]
        dim_concept = rnn_config["dim_concept"]
        dim_question = rnn_config["dim_question"]
        dim_correct = rnn_config["dim_correct"]
        dim_emb = dim_concept + dim_question + dim_correct
        dim_latent = self.params["models_config"]["kt_model"]["encoder_layer"]["dim_latent"]

        self.gru = nn.GRU(dim_emb + dim_latent, 128, batch_first=True)
        self.linear = nn.Linear(128, 1)

    def forward(self, x, z, padding):
        x = F.gelu(self.gru(torch.cat([x, z], dim=-1))[0])
        x = self.linear(x).squeeze(2)
        return (1.0 - padding.float()) * x
