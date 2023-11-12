import torch
import torch.nn as nn


from .AKT_Block import Architecture


class EncoderLayer(nn.Module):
    def __init__(self, params, objects):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.objects = objects

        self.encoder_type = params["models_config"]["kt_model"]["encoder_layer"]["type"]
        if self.encoder_type == "AKT":
            self.encoder = Architecture(self.params)
        else:
            raise NotImplementedError()

    def forward(self, batch):
        return self.encoder(batch)

    def get_latent(self, batch):
        return self.encoder.get_latent(batch)
