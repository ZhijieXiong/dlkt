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
            pass

    def forward(self):
        pass

    def get_latent(self, embeddings):
        if self.encoder_type == "AKT":
            pass


