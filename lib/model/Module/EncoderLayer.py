import torch
import torch.nn as nn


from .AKT_Block import Architecture


class EncoderLayer(nn.Module):
    def __init__(self, params, objects):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.objects = objects

        encoder_config = params["model_config"]["kt_model"]["encoder_layer"]
        self.encoder_type = encoder_config["type"]
        if self.encoder_type == "AKT":
            self.encoder = Architecture(self.params)
        else:
            # DKT
            rnn_type, dim_emb, dim_latent, rnn_layer = encoder_config["params"]
            if rnn_type == "rnn":
                self.encoder = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=rnn_layer)
            elif rnn_type == "lstm":
                self.encoder = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=rnn_layer)
            else:
                self.encoder = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=rnn_layer)

    def forward(self):
        pass

    def get_latent(self, embeddings):
        if self.encoder_type == "AKT":
            pass


