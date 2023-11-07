import torch.nn as nn

from .Module.KTEmbedLayer import KTEmbedLayer


class DKT(nn.Module):
    model_name = "DKT"

    def __init__(self, params, objects):
        super(DKT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = None
        self.encoder_layer = None
        self.predict_layer = None

        self.init_model()

    def init_model(self):
        self.embed_layer = KTEmbedLayer(self.params, self.objects)

        encoder_config = self.params["models_config"]["kt_model"]["encoder_layer"]
        dim_emb = encoder_config["dim_emb"]
        dim_latent = encoder_config["dim_latent"]
        rnn_type = encoder_config["rnn_type"]
        num_rnn_layer = encoder_config["num_rnn_layer"]
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)

        predict_layer_config = self.params["models_config"]["kt_model"]["predict_layer"]
        dropout = predict_layer_config["dropout"]
        num_predict_layer = predict_layer_config["num_predict_layer"]
        dim_predict_mid = predict_layer_config["dim_predict_mid"]
        activate_type = predict_layer_config["activate_type"]
        if activate_type == "tanh":
            act_func = nn.Tanh
        elif activate_type == "relu":
            act_func = nn.ReLU
        else:
            act_func = nn.Sigmoid
        dim_predict_out = predict_layer_config["dim_predict_out"]
        self.predict_layer = []
        if num_predict_layer == 1:
            self.predict_layer.append(nn.Dropout(dropout))
            self.predict_layer.append(nn.Linear(dim_latent, dim_predict_out))
            self.predict_layer.append(act_func())
        else:
            self.predict_layer.append(nn.Linear(dim_latent, dim_predict_mid))
            for _ in range(num_predict_layer - 1):
                self.predict_layer.append(act_func())
                self.predict_layer.append(nn.Dropout(dropout))
                self.predict_layer.append(nn.Linear(dim_predict_mid, dim_predict_mid))
            self.predict_layer.append(nn.Dropout(dropout))
            self.predict_layer.append(nn.Linear(dim_latent, dim_predict_out))
            self.predict_layer.append(act_func())

    def forward(self, batch):
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        interaction_seq = concept_seq[:, 0:-1].long() + self.model.num_item * correct_seq[:, 0:-1].long()
        emb_interaction = self.embed_layer.get_emb("interaction", interaction_seq)
        self.rnn.flatten_parameters()
        latent, _ = self.rnn(emb_interaction)
        predict_score = self.predict_layer(latent)

        return predict_score

    def get_latent(self, batch):
        correct_seq = batch["correct_seq"]
        concept_seq = batch["concept_seq"]
        interaction_seq = concept_seq[:, 0:-1].long() + self.model.num_item * correct_seq[:, 0:-1].long()
        emb_interaction = self.embed_layer.get_emb("interaction", interaction_seq)
        self.rnn.flatten_parameters()
        latent, _ = self.rnn(emb_interaction)

        return latent
